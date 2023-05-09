import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D, Orbitals, qmctorch_orbitals
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import qmctorch_nn_potential, qmctorch_en_potential, CoulombPairPotential
from VMC import GSVMC

from qmctorch.scf import Molecule
from qmctorch.wavefunction import SlaterJastrow

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
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")

    parser.add_argument("--iternum", type=int, default=10, help="number of new iterations")
    parser.add_argument("--batch", type=int, default=1000, help="batch size")

    parser.add_argument("--viz_flow", action="store_true", help="Visualize changing density in final flow")
    parser.add_argument("--viz_opt", action="store_true", help="Visualize final density in changing flow")
    parser.add_argument("--viz_bf", action="store_true", help="Visualize optimization of backflow potentials")
    parser.add_argument('--results_dir', type=str, default="./results")
    
    args = parser.parse_args()

    # device = torch.device("cuda:%d" % args.cuda)
    device = torch.device('cpu')
 
    # Define molecule, wavefunction, orbitals, nuclear potential
    mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', calculator='pyscf', basis='sto-3g', unit='bohr')
    wf = SlaterJastrow(mol)

    orbitals = qmctorch_orbitals()
    orbitals.orbitals = [one_orbital(i) for i in range(wf.nmo_opt)]  
    basedist = FreeFermion(device=device)

    # Initialize backflow for Continuous Normalizing Flow
    eta = MLP(1, args.Deta)
    eta.init_zeros()
    if not args.nomu:
        mu = MLP(1, args.Dmu)
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
    model.to(device=device)

    # Print some info
    print("nup = %d, ndown = %d, Z = %.1f" % (args.nup, args.ndown, args.Z))
    print("batch = %d, iternum = %d." % (args.batch, args.iternum))

    # Initialize equilibration energy plots
    fig1, (ax1_m, ax1_v) = plt.subplots(1,2, figsize=(12,8))
    fig1.suptitle('Energy during equilibration')
    ax1_m.set_title('mean')
    ax1_v.set_title('standard deviation')

    # Test equilibration time and stepsize
    model.equilibration_energy = True
    eq_steps = np.arange(1.5,4,0.5)
    eq_tau = np.arange(0.01, 0.05, 0.01)
    sp_mean_energy = np.ndarray((len(eq_steps),len(eq_tau)))
    sp_std_energy = np.ndarray((len(eq_steps),len(eq_tau)))
    for i, eq_s in enumerate(eq_steps):
        for j, eq_t in enumerate(eq_tau):
            model.equilibrium_steps = int(10**eq_s)
            model.tau = eq_t 
            gradE = model(args.batch)
            sp_mean_energy[i,j] = model.E
            sp_std_energy[i,j] = model.E_std
            E_vs_eq, Estd_vs_eq = np.squeeze(model.basedist.E_eq).T
            eq_step = np.log10(np.arange(1,model.equilibrium_steps+2))
            if j == 0:
                ax1_m.plot(eq_step, E_vs_eq, zorder=np.max(eq_steps)-i)
                ax1_v.plot(eq_step, Estd_vs_eq, zorder=np.max(eq_steps)-i)
    
    if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
    plt.savefig(os.path.join(args.results_dir, f"energy_during_equilibration.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
    plt.close()
    
    fig2, (ax2_steps, ax2_tau) = plt.subplots(1,2, figsize=(12,8))
    fig2.suptitle('Mean energy after equilibration')
    ax2_steps.set_title('vs. number of steps')
    ax2_tau.set_title('vs. step size')
    for i in range(np.shape(sp_mean_energy)[1]):
        ax2_steps.plot(eq_steps, sp_mean_energy[:,i], label=str(np.round(eq_tau[i],3)))
    for i in range(np.shape(sp_mean_energy)[0]):
        ax2_tau.plot(eq_tau, sp_mean_energy[i,:], label=str(np.round(eq_steps[i],1)))
    ax2_steps.legend()
    ax2_tau.legend()
    plt.savefig(os.path.join(args.results_dir, f"energy_after_equilibration.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
    plt.close()
