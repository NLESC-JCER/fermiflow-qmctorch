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
    def __init__(self, eta):
        """ The argument eta must be an instance of torch.nn.Module. """
        super(Backflow, self).__init__()
        self.eta = eta
        
    def _e_e(self, x):
        """
            The two-body part xi^{e-e}_i of the backflow transformation, which
        takes cares of the two-body correlation of the system. It reads as follows:
            xi^{e-e}_i = \\sum_{j neq i} eta(|r_i - r_j|) * (r_i - r_j).
        where eta is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 
        """
        _, n, dim = x.shape

        rij = x[:, :, None] - x[:, None]
        rij += torch.eye(n, device=x.device)[:, :, None]
        dij = rij.norm(dim=-1, keepdim=True)
        output = (self.eta(dij) * rij).sum(dim=-2)
        output -= self.eta(torch.ones(dim, device=x.device).norm()[None])
        return output

    def _e_e_divergence(self, x):
        """
            The divergence of the two-body part xi^{e-e}_i of the transformation, 
        which is derived and coded by hand to avoid the computational overhead in CNF. 
        The result is:
            div^{e-e} = \\sum_{i neq j}^{n} ( eta^prime(|r_i - r_j|) * |r_i - r_j|
                                        + dim * eta(|r_i - r_j|) ).
        where eta^prime denotes the derivative of the function eta, n is the total
        particle number, and dim is the space dimension.
        """
        _, n, dim = x.shape
        row_indices, col_indices = torch.triu_indices(n, n, offset=1)

        rij = x[:, :, None] - x[:, None]
        dij = rij.norm(dim=-1, keepdim=True)[:, row_indices, col_indices, :]
        eta, d_eta = self.eta(dij), self.eta.grad(dij)
        div_e_e = 2 * (d_eta * dij + dim * eta).sum(dim=(-2, -1))
        return div_e_e
    
    def forward(self, x):
        """
            The total backflow transformation xi_i, which contains the two-body part
        and (possibly) the one-body part:
            xi_i = xi^{e-e}_i + xi^{e-n}_i.

            It is easy to see that both components serve as equivariant functions 
        respect to any permutation of particle positions, then so do their sum.
        """
        return self._e_e(x)

    def divergence(self, x):
        """
            The divergence of the total backflow transformation, which contains the 
        two-body part and (possibly) the one-body part:
            div = div^{e-e} + div^{e-n}.
        """
        return self._e_e_divergence(x)

Deta = 50
eta = MLP(1, Deta)
# eta.init_zeros()
eta = lambda x: 0.8*torch.exp(-x/3)
v = Backflow(eta=eta)

Nbatch = 10
Npoints = 3

X   = 2*torch.randn((Nbatch,Npoints,3))
y1  = v(X).detach().numpy()

x = X.detach().numpy()

r = torch.linspace(0,10,100)[:,None]
ETA = v.eta(r)

fig = plt.figure(figsize=(12, 12), dpi=200)
plt.tight_layout()

ax1 = fig.add_subplot(111)
ax1.set_xlabel('r')
ax1.set_ylabel('eta')
ax1.grid()    
                
ax1.plot(r.detach().numpy(), ETA.detach().numpy(), color = 'tab:blue', zorder = 1)

plt.savefig(os.path.join("./results", f"two-body-flow-curve.jpg"),
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

pick1 = int(Nbatch*torch.rand((1,)).item())
pick2 = int(Npoints*torch.rand((1,)).item())

arrow_x = x[pick1,pick2,0].item()
arrow_y = x[pick1,pick2,1].item()
arrow_dx = y1[pick1,pick2,0].item() - arrow_x
arrow_dy = y1[pick1,pick2,1].item() - arrow_y

ax1.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, color='r')

plt.savefig(os.path.join("./results", f"two-body-flow-result.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
plt.close()

