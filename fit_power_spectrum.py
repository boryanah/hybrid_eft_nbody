import numpy as np
import os
from nbodykit.lab import *
import pyccl as ccl
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.compute_fields import load_fields
from tools.power_spectrum import get_all_cross_ps, get_mesh_list, predict_Pk, get_Pk_true
#from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget

simulation_code = 'gadget'#'abacus'

if simulation_code == 'abacus':
    # simulation name, see https://abacussummit.readthedocs.io/en/latest/simulations.html
    sim_name = "AbacusSummit_hugebase_c000_ph000"#small/AbacusSummit_small_c000_ph3046

    # load simulation information
    lagr_pos, pos_snap, halo_table, header = read_abacus(sim_name,z_nbody)
    pos_halo = halo_table['x_L2com']
    
    # parameter choices for simulation TODO: read from header
    ppd = 2304 # particle per dimension in the sim
    z_nbody = 1. # redshift where we measure power spectrum
    Lbox = 2000. # box size of the simulation [Mpc/h]

    # cosmological parameters # TODO: read from file
    h = 0.6736
    n_s = 0.9649
    Omega_b = 0.02237/h**2
    Omega_c = 0.12/h**2
    sigma8_m = 0.807952

    # load the cosmology # TODO: move into function
    cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

    dens_dir = "/mnt/alan1/boryanah/ICs/density.npy"# TODO: change
    data_dir = "BLABLA"

elif simulation_code == 'gadget':
    # simulation parameters
    Lbox = 175.#Mpc/h
    ppd = 256
    # select snapshot
    ind_snap = 0; z_nbody = 1.
    
    # cosmological parameters
    n_s = 0.96
    Omega_c = 0.655
    Omega_b = 0.045
    h = 0.7
    sigma8_m = 0.8
    
    # directory where the data is saved
    dens_dir = "data/density.npy"
    data_dir = "data/"

    # create CCL cosmology object
    cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

    # directory of simulation
    directory = "/mnt/gosling1/boryanah/small_box_damonge/"#"/global/cscratch1/sd/damonge/NbodySims/Sim256/"

    # find all files
    ic_fns = sorted(glob.glob(directory+"ic_*"))
    snap_fns = sorted(glob.glob(directory+"snap_*"))
    fof_fns = sorted(glob.glob(directory+"fof_*.fits"))

    print(ic_fns,snap_fns,fof_fns)
    
    # return position of the particles and halos
    lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)

print("Obtained positions and snapshots")

# user choices
N_dim = ppd # particle mesh size
interlaced = True
R_smooth = 4.

# obtain the truth
ks, Pk_true = get_Pk_true(pos_halo,N_dim,Lbox,interlaced)
Pk_true = Pk_true.astype(np.float64)
print("Computed true power spectrum")

# load errors computed through jackknifing
# TODO: make it part of the module
Pk_err = np.load("data/Pk_true_err.npy")

# covariance matrix
cov = np.diag(Pk_err)
cov[0,0] = 1.
icov = np.linalg.inv(cov)
icov[0,0] = 1.e6

# load the 5 fields
ones, delta, delta_sq, nabla_sq, s_sq = load_fields(cosmo,dens_dir,data_dir,R_smooth,N_dim,Lbox,z_nbody)
print("Loaded all fields")

# create mesh list for the 5 fields
mesh_list = get_mesh_list(pos_snap,ones,delta,delta_sq,nabla_sq,s_sq,lagr_pos,Lbox,N_dim,interlaced)
print("Obtained mesh lists for all fields")

# compute all 15 combinations
if os.path.exists(data_dir+"Pk_all.npy"):
    ks_all = np.load(data_dir+"ks_all.npy")
    Pk_all = np.load(data_dir+"Pk_all_real.npy")
    k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
else:
    ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list)
    np.save(data_dir+"ks_all.npy",ks_all)
    np.save(data_dir+"Pk_all.npy",Pk_all)
    np.save(data_dir+"Pk_all_real.npy",Pk_all.astype(np.float64))
    np.save(data_dir+"k_lengths.npy",k_lengths)
print("Loaded all templates")

def calculate_chi2(f_i):
    Pk = predict_Pk(f_i,ks_all,Pk_all,k_lengths)
    dPk = Pk-Pk_true

    chi2 = np.dot(dPk,np.dot(icov,dPk))
    print("chi2 = ",chi2)
    return chi2

# initial choice for bias parameters: F_i = {1,b_1,b_2,b_nabla,b_s}
b_1 = 1.2
b_2 = 0.4
b_nabla = 0.1
b_s = 0.2
f_params = np.array([b_1, b_2, b_s, b_nabla])

# minimize to get parameters
x0 = f_params
xtol = 1.e-6
res = minimize(calculate_chi2, x0, method='powell',\
               options={'xtol': xtol, 'disp': True})

f_best = res.x
print(f_best)
Pk_best = predict_Pk(f_best,ks_all,Pk_all,k_lengths)

plt.errorbar(ks,Pk_true,yerr=Pk_err,color='black',label='halo-halo',zorder=1)
plt.plot(ks,Pk_best,color='dodgerblue',label='EFT-Hybrid',zorder=2)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("figs/Pk_fit.png")
plt.show()
