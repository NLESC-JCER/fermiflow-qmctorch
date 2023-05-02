import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import HO, CoulombPairPotential
from VMC import GSVMC

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ground-state variational Monte Carlo simulation")

    parser.add_argument("--nup", type=int, default=1, help="number of spin-up electrons")
    parser.add_argument("--ndown", type=int, default=1, help="number of spin-down electrons")
    parser.add_argument("--Z", type=float, default=0.5, help="Coulomb interaction strength")

    parser.add_argument("--cuda", type=int, default=0, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")

    parser.add_argument("--iternum", type=int, default=1000, help="number of new iterations")
    parser.add_argument("--batch", type=int, default=8000, help="batch size")
    
    args = parser.parse_args()

    # device = torch.device("cuda:%d" % args.cuda)
    device = torch.device('cpu')
 
    mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', calculator='pyscf', basis='sto-3g', unit='bohr')
    wf = SlaterJastrow(mol)

    orbitals = [wf.pos2mo]      # pos2mo takes shape (batch, dim*N), while Fermiflow uses (batch, N, dim)
                                #   Somewhere we need pos = torch.cat([pos[:,i,:] for i in range(N)],1)
                                #   Moreover, in FF, orbitals is a list of orbitals to be filled with electrons
                                #   Now we only need one (1 up, 1 down).
    basedist = FreeFermion(device=device)

    eta = MLP(1, args.Deta)
    eta.init_zeros()
    if not args.nomu:
        mu = MLP(1, args.Dmu)
        mu.init_zeros()
    else:
        mu = None
    v = Backflow(eta, mu=mu)

    t_span = (args.t0, args.t1)

    cnf = CNF(v, t_span)

    sp_potential = wf.nuclear_potential     # In Fermiflow the actual potential is the method V(x) of class sp_potential,
                                            # viz. sp_potential takes instance sp_potential, and it is evaluated sp_potentail.V
                                            # What is the best policy?
    pair_potential = CoulombPairPotential(args.Z)

    model = GSVMC(args.nup, args.ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=sp_potential)
    model.to(device=device)
    print("nup = %d, ndown = %d, Z = %.1f" % (args.nup, args.ndown, args.Z))


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("batch = %d, iternum = %d." % (args.batch, args.iternum))

    import time
    for i in range(1, args.iternum + 1):
        start = time.time()

        gradE = model(args.batch)
        optimizer.zero_grad()
        gradE.backward()
        optimizer.step()

        speed = (time.time() - start) * 100 / 3600
        print("iter: %03d" % i, "E:", model.E, "E_std:", model.E_std, 
                "Instant speed (hours per 100 iters):", speed)
