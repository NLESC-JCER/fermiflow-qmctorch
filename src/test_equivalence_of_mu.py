import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

from MLP import MLP
import os

class Backflow(torch.nn.Module):
    """
        The backflow transformation that generates the collective coordinates
    {xi_i} from the original ones {r_i}, where i = 1, ..., n, n being the total 
    particle number, and both xi_i and r_i are dim-dimensional vectors, dim being 
    the space dimension.
    """
    def __init__(self, mu, old_mu=False, nuclear_positions=None):
        """ The argument eta must be an instance of torch.nn.Module. """
        super(Backflow, self).__init__()
        self.mu = mu
        if nuclear_positions is None:
            print("Warning: Backflow potential mu is provided, but nuclear positions are not",\
                  "revert to default: single nucleus at origin")
            nuclear_positions = [[0,0,0]]
        self.nucl_pos = torch.Tensor(nuclear_positions)
        self.old_mu = old_mu

    def _e_n(self, x):
        """
            The one-body (i.e., mean-field) part xi^{e-n}_i of the backflow 
        transformation, which takes cares of the interaction of one particle with
        some "nuclei" positions in the system, possibly arising from the nuclei in 
        a real molecule or harmonic trap in cold-atom systems, and so on.
            For simplicity, it is assumed that there is only one nucleus position
        at the origin. Then the transformation reads as follows:
            xi^{e-n}_i = mu(|r_i|) * r_i.
        where mu is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 
        """
        _, n, dim = x.shape

        rij = x[:,:,None]-self.nucl_pos[None,None,:]
        dij = rij.norm(dim=-1,keepdim=True)
        output = (self.mu(dij) * rij).sum(dim=-2)
        return output

    def _e_n_divergence(self, x):
        """
            The divergence of the one-body part xi^{e-n}_i of the transformation, 
        which is derived and coded by hand to avoid the computational overhead in CNF. 
        The result (for the simplified single-nucleus case) is:
            div^{e-n} = \\sum_{i=1}^{n} ( mu^prime(|r_i|) * |r_i| 
                                        + dim * mu(|r_i|) ).
        where mu^prime denotes the derivative of the function mu, n is the total
        particle number, and dim is the space dimension.
        """
        _, n, dim = x.shape
        row_indices, col_indices = torch.triu_indices(n, n, offset=-n)

        rij = x[:,:,None]-self.nucl_pos[None,None,:]
        dij = rij.norm(dim=-1, keepdim=True)[:, row_indices, col_indices, :]
        mu, d_mu = self.mu(dij), self.mu.grad(dij)
        div_e_n = ( d_mu * dij + dim * mu ).sum(dim=(-2, -1))
        return div_e_n

    def _e_n_old(self, x):
        """
            The one-body (i.e., mean-field) part xi^{e-n}_i of the backflow 
        transformation, which takes cares of the interaction of one particle with
        some "nuclei" positions in the system, possibly arising from the nuclei in 
        a real molecule or harmonic trap in cold-atom systems, and so on.
            For simplicity, it is assumed that there is only one nucleus position
        at the origin. Then the transformation reads as follows:
            xi^{e-n}_i = mu(|r_i|) * r_i.
        where mu is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 
        """
        di = x.norm(dim=-1, keepdim=True)
        return self.mu(di) * x

    def _e_n_divergence_old(self, x):
        """
            The divergence of the one-body part xi^{e-n}_i of the transformation, 
        which is derived and coded by hand to avoid the computational overhead in CNF. 
        The result (for the simplified single-nucleus case) is:
            div^{e-n} = \\sum_{i=1}^{n} ( mu^prime(|r_i|) * |r_i| 
                                        + dim * mu(|r_i|) ).
        where mu^prime denotes the derivative of the function mu, n is the total
        particle number, and dim is the space dimension.
        """
        dim = x.shape[-1]

        di = x.norm(dim=-1, keepdim=True)
        mu, d_mu = self.mu(di), self.mu.grad(di)
        div_e_n = ( d_mu * di + dim * mu ).sum(dim=(-2, -1))
        return div_e_n
    
    def forward(self, x):
        """
            The total backflow transformation xi_i, which contains the two-body part
        and (possibly) the one-body part:
            xi_i = xi^{e-e}_i + xi^{e-n}_i.

            It is easy to see that both components serve as equivariant functions 
        respect to any permutation of particle positions, then so do their sum.
        """
        return self._e_n_old(x) if self.old_mu else self._e_n(x)

    def divergence(self, x):
        """
            The divergence of the total backflow transformation, which contains the 
        two-body part and (possibly) the one-body part:
            div = div^{e-e} + div^{e-n}.
        """
        return self._e_n_divergence_old(x) if self.old_mu else self._e_n_divergence(x)

Dmu = 50
mu = MLP(1, Dmu)
#mu = lambda x: 0.8*torch.exp(-x/3)
nucl = [[0,0,0]]
v1 = Backflow(mu=mu, old_mu=False, nuclear_positions=nucl)
v2 = Backflow(mu=mu, old_mu=True, nuclear_positions=nucl)

Npoints = 50

X   = 2*torch.randn((1,Npoints,len(nucl[0])))
y1  = v1(X).detach().numpy()
y2  = v2(X).detach().numpy()

x = X.detach().numpy()

r = torch.linspace(0,10,100)[:,None]
MU = v1.mu(r)

fig = plt.figure(figsize=(12, 12), dpi=200)
plt.tight_layout()

ax1 = fig.add_subplot(111)
ax1.set_xlabel('r')
ax1.set_ylabel('mu')
ax1.grid()    
                
ax1.plot(r.detach().numpy(), MU.detach().numpy(), color = 'tab:blue', zorder = 1)

plt.savefig(os.path.join("./results", f"one-body-flow-curve.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(12, 12), dpi=200)
plt.tight_layout()

ax1 = fig.add_subplot(111)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid()    
                
ax1.scatter([0],[0], color = 'k', zorder = .5)               
ax1.scatter(x[...,0], x[...,1], color = 'tab:blue', zorder = 1)
ax1.scatter(y1[...,0], y1[...,1], color = 'k', marker='+', zorder = 2)
ax1.scatter(y2[...,0], y2[...,1], color = 'tab:orange', alpha=0.3, zorder = 3)

pick = int(Npoints*torch.rand((1,)).item())

arrow_x = x[:,pick,0].item()
arrow_y = x[:,pick,1].item()
arrow_dx = y1[:,pick,0].item() - arrow_x
arrow_dy = y1[:,pick,1].item() - arrow_y

ax1.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, color='r')

plt.savefig(os.path.join("./results", f"one-body-flow-result.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
plt.close()

