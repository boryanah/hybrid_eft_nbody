import numpy as np
import os
import pyccl as ccl
import glob
import matplotlib.pyplot as plt

from tools.compute_fields import load_fields
from tools.power_spectrum import get_all_cross_ps, get_mesh_list
from tools.read_abacus import read_abacus

machine = 'alan'
#machine = 'NERSC'

# user choices, see https://abacussummit.readthedocs.io/en/latest/simulations.html
sim_name = "AbacusSummit_hugebase_c000_ph000"
#small/AbacusSummit_small_c000_ph3046
interlaced = True
R_smooth = 2.
N_dim = 2304 # particle mesh size; usually ppd
ppd = 2304 # particle per dimension in the sim
z_nbody = 1. # redshift where we measure power spectrum
Lbox = 2000. # box size of the simulation [Mpc/h]

# load simulation information # todo: load per chunk
lagr_pos, pos_snap, halo_table, header = read_abacus(sim_name,z_nbody)
pos_halo = halo_table['x_L2com']

# convert [-Lbox/2.,Lbox/2.] to [0,Lbox]
pos_halo += Lbox/2.
pos_snap += Lbox/2.
lagr_pos += Lbox/2.

if machine == 'alan':
    data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    dens_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/"
elif machine == 'NERSC':
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/"

print("Obtained positions and snapshots")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# save data
#np.save(data_dir+"pos_ic.npy",lagr_pos)
np.save(data_dir+"pos_snap.npy",pos_snap)
np.save(data_dir+"pos_halo.npy",pos_halo)
del pos_halo

# cosmological parameters 
h = 0.6736
n_s = 0.9649
Omega_b = 0.02237/h**2
Omega_c = 0.12/h**2
sigma8_m = 0.807952

# load the cosmology
cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

# scale the density as suggested in Modi et al.
D_z = ccl.growth_factor(cosmo,1./(1+z_nbody))
    
# load the 5 fields
ones, delta, delta_sq, nabla_sq, s_sq = load_fields(cosmo,dens_dir,R_smooth,N_dim,Lbox,z_nbody)
print("Loaded all fields")

# todo: do a for loop over chunks

# scale the fields
delta *= D_z
delta_sq *= D_z
nabla_sq *= D_z
s_sq *= D_z

# create mesh list for the 5 fields
mesh_list = get_mesh_list(pos_snap,ones,delta,delta_sq,nabla_sq,s_sq,lagr_pos,Lbox,N_dim,interlaced)
print("Obtained mesh lists for all fields")

# compute all cross power spectra
ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list)
Pk_all = Pk_all.astype(np.float64)
np.save(data_dir+"ks_all.npy",ks_all)
np.save(data_dir+"Pk_all_real_%d.npy"%(int(R_smooth)),Pk_all)
np.save(data_dir+"k_lengths.npy",k_lengths)

print("Saved all templates")
