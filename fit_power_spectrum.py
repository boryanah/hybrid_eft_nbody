import numpy as np
import os
from nbodykit.lab import *
import pyccl as ccl
from scipy.optimize import minimize

from compute_tools import load_fields
from power_spectrum_tools import get_all_cross_ps, get_mesh_list, predict_Pk, get_Pk_true
from read_abacus import read_abacus
from read_gadget import read_gadget

simulation_code = 'gadget'#'abacus'

if simulation_code == 'abacus':
    # For more specs, see https://abacussummit.readthedocs.io/en/latest/simulations.html
    
    # simulation name
    sim_name = "AbacusSummit_hugebase_c000_ph000"#small/AbacusSummit_small_c000_ph3046

    # load simulation information
    lagr_pos, pos_snap, halo_table, header = read_abacus(sim_name,z_nbody)
    pos_halo = halo_table['x_L2com']
    
    # parameter choices for simulation TODO: read from header
    ppd = 2304 # particle per dimension in the sim
    z_ic = 49. # redshift of initial conditions
    z_nbody = 1. # redshift where we measure power spectrum
    Lbox = 2000. # box size of the simulation [Mpc/h]
    cpd = 455 # cells per dimension (internal to code, unrelated to CiC)

    # cosmological parameters # TODO: read from file
    H0 = 67.36
    h = H0/100.
    n_s = 0.9649
    omega_b = 0.02237
    omega_cdm = 0.12
    Omega_b = omega_b/h**2
    Omega_c = omega_cdm/h**2
    N_ur = 2.0328
    N_ncdm = 1.0
    sigma8_m = 0.807952
    sigma8_cb = 0.811355 

    # load the cosmology # TODO: move into function
    cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

    dens_dir = "/mnt/alan1/boryanah/ICs/density.npy"# TODO: change
    data_dir = "BLABLA"


    
elif simulation_code == 'gadget':
    # simulation parameters
    Lbox = 175.#Mpc/h
    ppd = 256

    # cosmological parameters
    n_s = 0.96
    Omega_c = 0.655
    Omega_b = 0.045
    h = 0.7
    sigma_8_m = 0.8
    
    # directory where the data is saved
    dens_dir = "data/density.npy"
    data_dir = "data/"

    # create CCL cosmology object
    cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

    # directory of simulation
    directory = "/global/cscratch1/sd/damonge/NbodySims/Sim256/"

    # find all files
    ic_fns = sorted(glob.glob(directory+"ic_*"))
    snap_fns = sorted(glob.glob(directory+"snap_*"))
    fof_fns = sorted(glob.glob(directory+"fof_*.fits"))
    
    # select snapshot
    ind_snap = 0; z_nbody = 1.

    # return position of the particles and halos
    lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)
    
# user choices
N_dim = ppd # particle mesh size
interlaced = True
R_smooth = 4.

# obtain the truth
ks, Pk_true = get_Pk_true(pos_halo,N_dim,Lbox,interlaced)

# covariance matrix TODO: compute as jackknife
cov = np.diag(np.ones(len(Pk_true))*1.e-3)
icov = np.linalg.inv(cov)

# load the 5 fields
ones, delta, delta_sq, nabla_sq, s_sq = load_fields(cosmo,dens_dir,data_dir,R_smooth,N_dim,Lbox,z_nbody)

# create mesh list for the 5 fields
mesh_list = get_mesh_list(pos_snap,ones,delta,delta_sq,nabla_sq,s_sq,lagr_pos,Lbox,N_dim,interlaced)

# compute all 15 combinations
ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list)

def calculate_chi2(f_i):
    Pk = predict_Pk(f_i,ks_all,Pk_all,k_lengths)
    dPk = Pk-Pk_true

    chi2 = np.dot(dPk,np.dot(icov,dPk))

    return chi2

# initial choice for bias parameters: F_i = {1,b_1,b_2,b_nabla,b_s}
b_1 = 1.2
b_2 = 0.4
b_nabla = 0.1
b_s = 0.2
f_params = [b_1, b_2, b_s, b_nabla]

# minimize to get parameters
x0 = f_params
xtol = 1.e-1
res = minimize(calculate_chi2, x0, method='powell',\
               options={'xtol': xtol, 'disp': True})

f_best = res.x
print(f_best)
