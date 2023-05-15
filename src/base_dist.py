import torch
torch.set_default_dtype(torch.float64)
from utils import y_grad_laplacian

class BaseDist(object):
    """ The base class of base (i.e., prior) distribution. """
    def __init__(self):
        pass
    def log_prob(self, x):
        pass
    def sample(self, sample_shape):
        pass

from slater import LogAbsSlaterDet, LogAbsSlaterDetMultStates

class FreeFermion(BaseDist):
    """ 
        This class serves to compute the log probability and sample the eigenstates
    of an non-interacting Fermion system, i.e., Slater determinants.

        For a non-interacting systems with nup spin-up electrons and ndown spin-down
    electrons, any eigenstate wavefunction (after eliminating the spin indices) 
    is written as the product of spin-up and spin-down Slater determinants:
        \Psi(r^up_1, ..., r^up_nup, r^down_1, r^down_ndown)
         = det(\phi^up_j(r^up_i)) * det(\phi^down_j(r^down_i)),
    where \phi^up_j (j = 1, ..., nup), \phi^down_j (j = 1, ..., ndown) are the occupied
    single-particle orbitals for the spin-up and spin-down electrons, respectively.
    These orbitals are passed as arguments "orbitals_up" and "orbitals_down" in the
    class methods.

        Note the wavefunction above is not normalized. The normalization factor is 
    1 / sqrt(nup! * ndown!).

    ================================================================================
    Below are a diagram demonstrating dependencies among the various functions.
    "1 --> 2" indicates function 2 depends on function 1. 

    LogAbsSlaterDet --> log_prob --> sample --> sample_multstates_old
                           |
                           v
             log_prob_multstates (method 1) --> sample_multstates (method 1)

    LogAbsSlaterDetMultStates --> log_prob_multstates (method 2) --> sample_multstates (method 2)
    """

    def __init__(self, device=torch.device("cpu")):
        super(FreeFermion, self).__init__()
        self.device = device

    def log_prob(self, orbitals_up, orbitals_down, x):
        nup, ndown = len(orbitals_up), len(orbitals_down)
        logabspsi = (LogAbsSlaterDet.apply(orbitals_up, x[..., :nup, :]) 
                        if nup != 0 else 0) \
                  + (LogAbsSlaterDet.apply(orbitals_down, x[..., nup:, :])
                        if ndown != 0 else 0)
        logp = 2 * logabspsi
        return logp

    def get_energy(self, orbitals_up, orbitals_down, x, pot_ee, pot_en, pot_nn):
        x.requires_grad_(True)
        f = lambda x: self.log_prob(orbitals_up, orbitals_down, x)
        logp, grad_logp, laplacian_logp = y_grad_laplacian(f, x) 
        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))
        
        potential = pot_ee.V(x)
        if pot_en:
            potential += pot_en.V(x)
        if pot_nn:
            potential += pot_nn.V

        x.requires_grad_(False)

        Eloc = (kinetic + potential).detach()
        E, E_std = Eloc.mean().item(), Eloc.std().item()
        return E, E_std
    
    def sample(self, orbitals_up, orbitals_down, sample_shape, 
            equilibrim_steps=100, tau=0.1, equilibration_energy=False, pot_ee=None, pot_en=None, pot_nn=None):
        #print("Sample a Slater determinant...")
        nup, ndown = len(orbitals_up), len(orbitals_down)
        x = torch.randn(*sample_shape, nup + ndown, 2, device=self.device)
        logp = self.log_prob(orbitals_up, orbitals_down, x)
        self.E_eq = None
        if equilibration_energy:
            self.E_eq = [[self.get_energy(orbitals_up, orbitals_down, x, pot_ee, pot_en, pot_nn)]]
        
        for _ in range(equilibrim_steps):
            new_x = x + tau * torch.randn_like(x)
            new_logp = self.log_prob(orbitals_up, orbitals_down, new_x)
            p_accept = torch.exp(new_logp - logp)
            accept = torch.rand_like(p_accept) < p_accept
            x[accept] = new_x[accept]
            logp[accept] = new_logp[accept]
            if equilibration_energy:
                self.E_eq.append([self.get_energy(orbitals_up, orbitals_down, x, pot_ee, pot_en, pot_nn)])

        return x

    def log_prob_multstates(self, states, state_indices_collection, x, method=2):
        if len(x.shape[:-2]) != 1:
            raise ValueError("FreeFermion.log_prob_multstates: x is required to have "
                    "only one batch dimension.")
        if method == 1:

            """ Making use of log_prob. """

            batch = x.shape[0]
            logp = torch.empty(batch, device=x.device)
            base_idx = 0
            for idx, times in state_indices_collection.items():
                logp[base_idx:base_idx+times] = \
                    self.log_prob(*states[idx], x[base_idx:base_idx+times, ...])
                base_idx += times
            return logp
        elif method == 2:

            """ Making use of the LogAbsSlaterDetMultStates primitive. """

            states_up, states_down = tuple(zip(*states))
            nup, ndown = len(states_up[0]), len(states_down[0])
            logabspsi = (LogAbsSlaterDetMultStates.apply(states_up, state_indices_collection, x[..., :nup, :]) 
                            if nup != 0 else 0) \
                      + (LogAbsSlaterDetMultStates.apply(states_down, state_indices_collection, x[..., nup:, :])
                            if ndown != 0 else 0)
            logp = 2 * logabspsi
            return logp

    def sample_multstates(self, states, state_indices_collection, sample_shape, 
            equilibrim_steps=100, tau=0.1, cpu=False, method=2):
        if len(sample_shape) != 1:
            raise ValueError("FreeFermion.sample_multstates: sample_shape is "
                    "required to have only one batch dimension.")

        #import time
        nup, ndown = len(states[0][0]), len(states[0][1])
        x = torch.randn(*sample_shape, nup + ndown, 2, 
                        device=torch.device("cpu") if cpu else self.device)
        logp = self.log_prob_multstates(states, state_indices_collection, x, method=method)
        #print("x.device:", x.device, "logp.device:", logp.device, "method:", method)

        for _ in range(equilibrim_steps):
            #start_out = time.time()

            new_x = x + tau * torch.randn_like(x)

            #start_in = time.time()
            new_logp = self.log_prob_multstates(states, state_indices_collection, new_x, method=method)
            #t_in = time.time() - start_in

            p_accept = torch.exp(new_logp - logp)
            accept = torch.rand_like(p_accept) < p_accept
            x[accept] = new_x[accept]
            logp[accept] = new_logp[accept]

            #t_out = time.time() - start_out
            #print("t_out:", t_out, "t_in:", t_in, "t_remain:", t_out - t_in, "ratio:", t_in / t_out)

        if cpu:
            x = x.to(device=self.device)
        return x

    def sample_multstates_old(self, states, state_indices_collection, sample_shape, 
            equilibrim_steps=100, tau=0.1):
        if len(sample_shape) != 1:
            raise ValueError("FreeFermion.sample_multstates_old: sample_shape is "
                    "required to have only one batch dimension.")

        xs = tuple( self.sample(*states[idx], (times,), 
                    equilibrim_steps=equilibrim_steps, tau=tau)
                for idx, times in state_indices_collection.items() )
        x = torch.cat(xs, dim=0)
        return x
