import torch
torch.set_default_dtype(torch.float64)

class GSVMC(torch.nn.Module):
    def __init__(self, nup, ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=None):
        """
            Ground State Variational Monte Carlo calculation.

        ---- INPUT ARGUMENTS ----

        nup, ndown: the number of spin-up and spin-down electrons.

        orbitals, basedist: orbitals contains the information of single-particle 
            orbitals and, combined with basedist, completely characterizes the base 
            distribution of the flow model.

        cnf: Continuous normalizing flow, which is an instance of the class CNF.
        """
        super(GSVMC, self).__init__()

        self.orbitals_up, self.orbitals_down = orbitals.orbitals[:nup], \
                                               orbitals.orbitals[:ndown]
        self.basedist = basedist
        self.cnf = cnf

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential
        self.equilibration_energy = False
        self.equilibrium_steps = 100
        self.tau = 0.1

    def sample(self, sample_shape):
        z = self.basedist.sample(self.orbitals_up, self.orbitals_down, sample_shape,\
                                    self.equilibrium_steps, self.tau,\
                                    equilibration_energy=self.equilibration_energy,\
                                    pot_ee=self.pair_potential, pot_en=self.sp_potential)
        x = self.cnf.generate(z)
        return z, x

    def logp(self, x, params_require_grad=False):
        z, delta_logp = self.cnf.delta_logp(x, params_require_grad=params_require_grad)
        logp = self.basedist.log_prob(self.orbitals_up, self.orbitals_down, z) - delta_logp
        return logp

    def forward(self, batch):
        from utils import y_grad_laplacian

        _, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = (kinetic + potential).detach()

        self.E, self.E_std = Eloc.mean().item(), Eloc.std().item()
        gradE = (logp_full * (Eloc.detach() - self.E)).mean()
        return gradE

class BetaVMC(torch.nn.Module):
    def __init__(self, beta, nup, ndown, deltaE, boltzmann,
                 orbitals, basedist, cnf, pair_potential, sp_potential=None):
        """
            Finite temperature Variational Monte Carlo calculation.

        ---- NOTABLE ARGUMENTS ----
            
        deltaE: The maximum excitation energy of the truncated states. In the present
            implementation, the case of Fermions trapped in 2D harmonic potential is
            considered, and deltaE takes value up to 4. See orbitals.py for details.
        """
        super(BetaVMC, self).__init__()

        self.beta = beta
        self.states, self.Es_original = orbitals.fermion_states(nup, ndown, deltaE)
        self.Es_original = torch.tensor(self.Es_original, dtype=torch.float64)
        self.Nstates = len(self.states)
        self.log_state_weights = torch.nn.Parameter(
                -self.beta * (self.Es_original - self.Es_original[0])
                if boltzmann else torch.randn(self.Nstates))

        self.basedist = basedist
        self.cnf = cnf

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential

    def sample(self, sample_shape, nframes=None):
        from torch.distributions.categorical import Categorical
        from collections import Counter
        import time

        self.state_dist = Categorical(logits=self.log_state_weights)
        state_indices = self.state_dist.sample(sample_shape)
        self.state_indices_collection = Counter(sorted(state_indices.tolist()))

        start = time.time()
        z = self.basedist.sample_multstates(self.states, 
                self.state_indices_collection, sample_shape)
        print("Finished sampling basis states. Time to take (hours per 100 iters):", 
                (time.time() - start) * 100 / 3600)

        x = self.cnf.generate(z, nframes=nframes)
        return z, x 

    def logp(self, x, params_require_grad=False):
        z, delta_logp = self.cnf.delta_logp(x, params_require_grad=params_require_grad)

        log_prob_z = self.basedist.log_prob_multstates(self.states, 
                self.state_indices_collection, z)

        logp = log_prob_z - delta_logp
        return logp

    def forward(self, batch):
        """
            Physical quantities of interest:
        self.E, self.E_std: mean and standard deviation of energy.
        self.F, self.F_std: mean and standard deviation of free energy.
        self.S, self.S_analytical: entropy of the system, computed using Monte Carlo
            sampling and the direct formula of von-Neumann, respectively.
        self.logp_states_all: lop-probability of each of the considered states, 
            which is represented by a 1D tensor of size self.Nstates.
        """
        from utils import y_grad_laplacian
        import time

        _, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        start = time.time()
        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        print("Computed gradients of logp up to 2nd order. "
                "Time to take (hours per 100 iters):", 
                (time.time() - start) * 100 / 3600)

        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = (kinetic + potential).detach()
        self.E, self.E_std = Eloc.mean().item(), Eloc.std().item()

        state_indices = torch.tensor(list(self.state_indices_collection.elements()), 
                            device=x.device)
        logp_states = self.state_dist.log_prob(state_indices)

        Floc = Eloc + logp_states.detach() / self.beta
        self.F, self.F_std = Floc.mean().item(), Floc.std().item()

        self.S = -logp_states.detach().mean().item()
        self.logp_states_all = self.state_dist.log_prob(torch.arange(self.Nstates, 
                            device=x.device)).detach()
        self.S_analytical = -(self.logp_states_all * 
                              self.logp_states_all.exp()).sum().item()

        gradF_phi = (logp_states * (Floc - self.F)).mean()

        Eloc_x_mean = torch.empty_like(Eloc)
        base_idx = 0
        for idx, times in self.state_indices_collection.items():
            Eloc_x_mean[base_idx:base_idx+times] = Eloc[base_idx:base_idx+times].mean().expand(times)
            base_idx += times
        gradF_theta = (logp_full * (Eloc - Eloc_x_mean)).mean()

        return gradF_phi, gradF_theta
