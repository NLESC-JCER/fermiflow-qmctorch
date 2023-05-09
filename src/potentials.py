import numpy as np
import torch
torch.set_default_dtype(torch.float64)

class SPPotential(object):
    def __init__(self):
        pass

class HO(SPPotential):
    def __init__(self):
        super(SPPotential, self).__init__()

    def V(self, x):
        return 0.5 * (x**2).sum(dim=(-2, -1))
    
class qmctorch_en_potential(SPPotential):
    def __init__(self, nuclear_potential):
        super(qmctorch_en_potential, self).__init__()
        self.nuclear_potential = nuclear_potential

    def V(self, x):
        Nbatch, N, dim = x.shape
        # reshape FF pos to QMCT pos: (Nbatch, N, dim) -> (Nbatch, N*dim) where dim is faster (x1,y1,z1,x2,...,zN)
        x = x.view(Nbatch, N*dim)
        # Pass pos to nuclear_potential
        v_en = self.nuclear_potential(x)
        # N.B. output nuclear_potential already shape (Nbatch,1)
        return v_en.squeeze()

class qmctorch_nn_potential(SPPotential):
    def __init__(self, mol):
        super(qmctorch_nn_potential, self).__init__()
        # For fixed geometry, compute term once
        self.natom = mol.natom
        self.coords = torch.Tensor(mol.atom_coords)
        self.charge = torch.Tensor(mol.atomic_number)
        self.dij = (self.coords[:,None]-self.coords[None]).norm(dim=-1)
        self.ZZ = self.charge.view(-1,1)@self.charge.view(1,-1)
        self.V = 0
        for i in range(self.natom):
            for j in range(i+1, self.natom):
                self.V += self.ZZ[i,j].item()/self.dij[i,j].item()

# ==================================================================================

class PairPotential(object):
    
    def __init__(self):
        pass

    def rij(self, x):
        """
            Compute |r_i - r_j| for all 1 <= i < j <= n, n being the particle number.
            Input shape: (batch, n, dim)
            Output shape: (batch, n(n-1)/2)
        """
        n = x.shape[-2]
        row_indices, col_indices = torch.triu_indices(n, n, offset=1)
        return (x[:, :, None] - x[:, None])[:, row_indices, col_indices, :].norm(dim=-1)

    def V(self, x):
        """
            Compute the total potential energy V from the pair-wise potential function v:
            V = \sum_{i<j}^n v(|r_i - r_j|).
        """
        rij = self.rij(x)
        return self.v(rij).sum(dim=-1)

class CoulombPairPotential(PairPotential):
    def __init__(self, Z):
        super(CoulombPairPotential, self).__init__()
        self.Z = Z

    def v(self, rij):
        return self.Z / rij
