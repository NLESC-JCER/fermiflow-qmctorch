import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D
from base_dist import FreeFermion

from MLPextended import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import HO, CoulombPairPotential
from VMC import GSVMC

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ground-state variational Monte Carlo simulation")

    parser.add_argument("--nup", type=int, default=6, help="number of spin-up electrons")
    parser.add_argument("--ndown", type=int, default=0, help="number of spin-down electrons")
    parser.add_argument("--Z", type=float, default=0.5, help="Coulomb interaction strength")

    parser.add_argument("--cuda", type=int, default=0, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1, help="ending time")

    parser.add_argument("--iternum", type=int, default=10, help="number of new iterations")
    parser.add_argument("--batch", type=int, default=8000, help="batch size")
    parser.add_argument('--results_dir', type=str, default="./results")
    
    args = parser.parse_args()

    # device = torch.device("cuda:%d" % args.cuda)
    device = torch.device("cpu")

    orbitals = HO2D()
    basedist = FreeFermion(device=device)

    eta = MLP([1, 25, 25])
    eta.init_zeros()
    if not args.nomu:
        mu = MLP([1, 25, 25])
        mu.init_zeros()
    else:
        mu = None
    v = Backflow(eta, mu=mu)

    t_span = (args.t0, args.t1)

    cnf = CNF(v, t_span)

    sp_potential = HO()
    pair_potential = CoulombPairPotential(args.Z)

    model = GSVMC(args.nup, args.ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=sp_potential)
    model.to(device=device)
    print("nup = %d, ndown = %d, Z = %.1f" % (args.nup, args.ndown, args.Z))


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("batch = %d, iternum = %d." % (args.batch, args.iternum))

    import time
    mean_E = np.ndarray((args.iternum+1,))
    std_E = np.ndarray((args.iternum+1,))
    model.equilibration_energy = True
    gradE = model(args.batch)
    
    mean_E[0], std_E[0] = model.E, model.E_std
    equil_before_opt = np.squeeze(model.basedist.E_eq)

    model.equilibration_energy = False
    for i in range(1, args.iternum + 1):
        start = time.time()

        gradE = model(args.batch)
        optimizer.zero_grad()
        gradE.backward()
        optimizer.step()

        speed = (time.time() - start) * 100 / 3600
        print("iter: %03d" % i, "E:", model.E, "E_std:", model.E_std, 
                "Instant speed (hours per 100 iters):", speed)
    
        mean_E[i], std_E[i] = model.E, model.E_std

    model.equilibration_energy = True
    gradE = model(args.batch)
    equil_after_opt = np.squeeze(model.basedist.E_eq)
    model.equilibration_energy = False

    var_E = std_E**2
    it = np.arange(args.iternum+1)
    np.savetxt('energy_variance.txt', np.vstack((it, mean_E, var_E)).T, fmt=['%d', '%.3f', '%.3f'], header='iteration - energy - variance')
    np.savetxt('energy_equilibration.txt', np.hstack((equil_before_opt, equil_after_opt)), fmt='%.3f')
    
    r_bf = torch.linspace(0,20,200)[:,None]
    eta_r = model.cnf.v_wrapper.v.eta(r_bf)
    if not args.nomu:
        mu_r = model.cnf.v_wrapper.v.mu(r_bf)
    else:
        mu_r = torch.zeros_like(eta_r)

    fig = plt.figure(figsize=(12, 8), dpi=200)
    plt.tight_layout()

    ax1 = fig.add_subplot(121)
    ax1.set_title('electron-electron')
    ax1.set_xlim(0, 20)
    ax1.set_xlabel('$r$')
    ax1.set_ylabel(u'\u03B7($r$)')
    ax1.grid()
                
    ax1.plot(r_bf.detach().cpu().numpy(), eta_r.detach().cpu().numpy())

    ax2 = fig.add_subplot(122, sharey = ax1)
    ax2.set_title('electron-nucleus')
    ax2.set_xlim(0, 20)
    ax2.set_xlabel('$r$')
    ax2.set_ylabel(u'\u03BC($r$)')
    ax2.grid()
                
    ax2.plot(r_bf.detach().cpu().numpy(), mu_r.detach().cpu().numpy())

    plt.savefig(os.path.join(args.results_dir, f"ho2d-backflow-viz.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
    plt.close()

    scale_var = 50
    fig = plt.figure(figsize=(12, 8), dpi=200)
    plt.tight_layout()

    ax1 = fig.add_subplot(111)
    ax1.set_xlim(-1, args.iternum+1)
    ax1.set_title("Average energy each iteration with indication of variance")
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('energy')
    ax1.grid()    
                
    ax1.fill_between(np.arange(args.iternum + 1), mean_E + var_E/scale_var, mean_E - var_E/scale_var, alpha = 0.2, color = 'C0', zorder = 1)
    ax1.plot(np.arange(args.iternum + 1), mean_E - var_E/scale_var, color = 'tab:blue', zorder = 2)
    ax1.plot(np.arange(args.iternum + 1), mean_E + var_E/scale_var, color = 'tab:blue', zorder = 3)
    ax1.plot(np.arange(args.iternum + 1), mean_E, color = 'r', zorder = 4)

    ax1.hlines([18.179], xmin=-1, xmax=args.iternum + 1, colors='k', linestyles='--', zorder = 3.5)

    plt.savefig(os.path.join(args.results_dir, f"ho2d-energy-iterations.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
    plt.close()
