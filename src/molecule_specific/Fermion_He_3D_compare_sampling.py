import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D, Orbitals, qmctorch_orbitals
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import qmctorch_en_potential, qmctorch_nn_potential, CoulombPairPotential
from VMC import GSVMC

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow
from qmctorch.sampler import Metropolis
from qmctorch.solver import Solver

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from mendeleev import element

def one_orbital(orb_idx):
    def orbital(x):
        Nbatch, N, dim = x.shape
        # reshape FF pos to QMCT pos: (Nbatch, N, dim) -> (Nbatch, N*dim) where dim is faster (x1,y1,z1,x2,...,zN)
        x = x.view(Nbatch, N*dim)
        mo = wf.pos2mo(x)
        mo = mo.view(Nbatch, N, -1)
        return mo[..., orb_idx]
    return orbital

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ground-state variational Monte Carlo simulation")

    parser.add_argument("--nup", type=int, default=1, help="number of spin-up electrons")
    parser.add_argument("--ndown", type=int, default=1, help="number of spin-down electrons")
    parser.add_argument("--Z", type=float, default=1, help="Coulomb interaction strength")

    parser.add_argument("--cuda", type=int, default=0, help="GPU device number")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")

    parser.add_argument("--iternum", type=int, default=100, help="number of new iterations")
    parser.add_argument("--return_last", type=float, default=0.1, help="Fraction of iterations over which to compute average")
    parser.add_argument("--batch", type=int, default=1000, help="batch size")

    parser.add_argument("--viz_flow", action="store_true", help="Visualize changing density in final flow")
    parser.add_argument("--viz_bf", action="store_true", help="Visualize optimization of backflow potentials")
    parser.add_argument('--results_dir', type=str, default="./results")
    
    args = parser.parse_args()

    # device = torch.device("cuda:%d" % args.cuda)
    device = torch.device('cpu')
 
    # Define molecule, wavefunction, orbitals, nuclear potential
    mol = Molecule(atom='He 0 0 0', calculator='pyscf', basis='sto-3g', unit='bohr')
    wf = SlaterJastrow(mol).gto2sto()

    orbitals = qmctorch_orbitals()
    orbitals.orbitals = [one_orbital(i) for i in range(wf.nmo_opt)]  
    basedist = FreeFermion(device=device)

    # Initialize backflow for Continuous Normalizing Flow
    eta = MLP(1, 50)
    eta.init_zeros()
    if not args.nomu:
        mu = MLP(1, 50)
        mu.init_zeros()
    else:
        mu = None
    v = Backflow(eta, mu=mu, nuclear_positions=mol.atom_coords)

    # Build model and initialize optimizer
    t_span = (args.t0, args.t1)
    cnf = CNF(v, t_span)

    pair_potential = CoulombPairPotential(args.Z)               # e-e potential with strength Z
    sp_potential = qmctorch_en_potential(wf.nuclear_potential)  # e-n potential
    nucl_potential = qmctorch_nn_potential(wf.mol)              # n-n potential

    model = GSVMC(args.nup, args.ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=sp_potential, nucl_potential=nucl_potential)
    model.equilibrium_steps = 500
    model.tau = 0.1
    model.to(device=device)

    model.equilibration_energy = True
    gradE = model(args.batch)

    equil_from_FermiFlow = np.squeeze(model.basedist.E_eq)

    domain = mol.domain('normal')
    domain['mean'] = np.zeros(wf.ndim)
    domain['sigma'] = np.eye(wf.ndim)
    sampler = Metropolis(nwalkers=args.batch, nstep=model.equilibrium_steps, step_size=model.tau,
                     ntherm=0, ndecor=1, nelec=wf.nelec, ndim=wf.ndim,
                     init=domain,
                     move={'type': 'all-elec', 'proba': 'normal'})
    solver = Solver(wf=wf, sampler=sampler)

    pos = solver.sampler(solver.wf.pdf)
    # OR
    # pos = torch.randn((1000,2,3))
    obs = solver.sampling_traj(pos)
    eloc = obs.local_energy

    equil_from_QMCTorch = np.stack([np.mean(eloc, axis=1), np.std(eloc, axis=1)], axis=1)

    np.savetxt('energy_equilibration_FF.txt', equil_from_FermiFlow, fmt='%.3f')
    np.savetxt('energy_equilibration_QT.txt', equil_from_QMCTorch, fmt='%.3f')
