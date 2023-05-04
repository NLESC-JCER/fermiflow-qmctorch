import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D, Orbitals
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import HO, qmctorch_pot, CoulombPairPotential
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
    parser.add_argument("--Z", type=float, default=0.5, help="Coulomb interaction strength")

    parser.add_argument("--cuda", type=int, default=0, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")

    parser.add_argument("--iternum", type=int, default=10, help="number of new iterations")
    parser.add_argument("--batch", type=int, default=80, help="batch size")

    parser.add_argument("--viz", action="store_true", help="Visualize results")
    parser.add_argument('--results_dir', type=str, default="./results")
    
    args = parser.parse_args()

    # device = torch.device("cuda:%d" % args.cuda)
    device = torch.device('cpu')
 
    mol = Molecule(atom='H 0 0 -0.69; H 0 0 0.69', calculator='pyscf', basis='sto-3g', unit='bohr')
    wf = SlaterJastrow(mol)

    orbitals = Orbitals()
    orbitals.orbitals = [one_orbital(i) for i in range(wf.nmo_opt)]  
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

    sp_potential = qmctorch_pot(wf.nuclear_potential)
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
 
    if args.viz:
        viz_samples = 3000
        viz_timesteps = 21

        print('Start vizualisation of evolution of', viz_samples, 'samples in', viz_timesteps, 'timesteps')
        
        # Relevant data for plotting of molecule
        atom_names_coords = [a.split() for a in mol.atoms_str.split(';')]
        atom_names = [a[0] for a in atom_names_coords]
        atom_coords =  np.array([[float(c) for c in a[1:]] for a in atom_names_coords])
        x_atoms, y_atoms, z_atoms = atom_coords[:,0], atom_coords[:,1], atom_coords[:,2]
        radii = [element(a).atomic_radius*0.0189 for a in atom_names]       # attribute in pm, need bohr

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            q_samples = model.basedist.sample(model.orbitals_up, model.orbitals_down, (viz_samples,))
            q_t_samples = model.cnf.generate(q_samples, nframes=viz_timesteps)

            res = 40
            # Generate evolution of density
            q_x = np.linspace(-1.5, 1.5, res)
            q_y = np.linspace(-1.5, 1.5, res)
            q_z = np.linspace(-1.5, 1.5, res)
            points = np.vstack(np.meshgrid(q_x, q_y, q_z)).reshape([3, -1]).T
            points = np.expand_dims(points, axis=1)

            q_density = torch.tensor(points).to(device)
            logp_diff = torch.zeros(q_density.shape[0], 1).to(device)
 
            q_t_density = model.cnf.generate(q_density, nframes=viz_timesteps)
            
            # q_t_(samples/density) of shape (viz_timesteps, ...), so each time step
            #   should be projected on plane (sum over 1 of the 3 remaining axes).

            # For the logp: coordinates go like (((for z in zs) for x in xs) for y in ys).
            #   So after exponent:  py = [p[i::10000].sum() for i in range(10000)]
            #       (for res=100)   px = [p[i:i+10000:100].sum() for i in range(0, 1000000, 10000)]
            #                       pz = [p[i:i+100:1].sum() for i in range(0, 1000000, 100)]
            
            # Create plots for each timestep
            for (t, q_sample, q_density) in zip(
                    np.linspace(args.t0, args.t1, viz_timesteps),
                    q_t_samples, q_t_density
            ):
                print('Create plot for time', np.round(t,3), 'out of', args.t1)
                fig = plt.figure(figsize=(12, 8), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}')

                ax1 = fig.add_subplot(2, 3, 1)
                ax1.set_title('Molecule')
                ax1.set_xlim(-1.5, 1.5)
                ax1.set_ylim(-1.5, 1.5)
                ax1.set_ylabel('XZ-plane')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(2, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(2, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])
                ax4 = fig.add_subplot(2, 3, 4)
                ax4.set_xlim(-1.5, 1.5)
                ax4.set_ylim(-1.5, 1.5)
                ax4.set_ylabel('XY-plane')
                ax4.get_xaxis().set_ticks([])
                ax4.get_yaxis().set_ticks([])
                ax5 = fig.add_subplot(2, 3, 5)
                ax5.get_xaxis().set_ticks([])
                ax5.get_yaxis().set_ticks([])
                ax6 = fig.add_subplot(2, 3, 6)
                ax6.get_xaxis().set_ticks([])
                ax6.get_yaxis().set_ticks([])
                
                # Calculate density at time t and projections on each plane
                model.cnf.t_span_reverse = args.t1, t
                if not t<args.t1:
                    model.cnf.t_span_reverse = args.t1, args.t1-(args.t1-args.t0)/viz_timesteps/100
                
                _, logp_diff = model.cnf.delta_logp(q_density)
                logp = model.basedist.log_prob(model.orbitals_up, model.orbitals_down, q_density) - logp_diff.view(-1)
                
                p = np.exp(logp.detach().cpu().numpy())

                print(logp.shape)
                print(p.shape)

                pXZ = np.array([p[i::res**2].sum() for i in range(res**2)])                     # project on xz (sum over y)
                pYZ = np.array([[p[i:i+res**2:res].sum() for i in range(j, res**3, res**2)] for j in range(res)]).flatten()     # project on yz (sum over x)
                pXY = np.array([p[i:i+res:1].sum() for i in range(0, res**3, res)])             # project on xy (sum over z)

                print(pXZ.shape, pYZ.shape, pXY.shape)

                # Everything in XZ-plane
                for x,y,r in zip(x_atoms, z_atoms, radii):
                    circle = plt.Circle((x, y), r, color='tab:blue') ; ax1.add_patch(circle)

                ax2.hist2d(q_sample.detach().cpu().numpy().T[0].flatten(), q_sample.detach().cpu().numpy().T[2].flatten(),
                           bins=200, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax3.tricontourf(*np.vstack(np.meshgrid(q_z, q_x)[::-1]).reshape([2, -1]), pXZ, 200)     # q_x & q_z switched around 
                                                                                                        #   as rescaling suggested
                                                                                                        #   q_x was not along x-axis,
                                                                                                        #   as it is in XY-plane plot.                               
                #  for yz-plane: tricontourf(*np.vstack(np.meshgrid(q_z, q_y)).reshape([2, -1]), pYZ, 200)
                
                # Everything in XY-plane
                for x,y,r in zip(x_atoms, y_atoms, radii):
                    circle = plt.Circle((x, y), r, color='tab:blue') ; ax4.add_patch(circle)

                ax5.hist2d(q_sample.detach().cpu().numpy().T[0].flatten(), q_sample.detach().cpu().numpy().T[1].flatten(),
                           bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])
                
                ax6.tricontourf(*np.vstack(np.meshgrid(q_x, q_y)).reshape([2, -1]),
                                pXY, 200)
                
                plt.savefig(os.path.join(args.results_dir, f"h2-cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"h2-cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "h2-cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "h2-cnf-viz.gif")))
