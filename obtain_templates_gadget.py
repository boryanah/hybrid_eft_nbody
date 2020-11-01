import numpy as np
import os
import pyccl as ccl
import glob
import matplotlib.pyplot as plt

from tools.compute_fields import load_fields
from tools.power_spectrum import get_all_cross_ps, get_mesh_list
from tools.read_gadget import read_gadget

machine = 'alan'
#machine = 'NERSC'

# user choices
sim_name = 'Sim256'
interlaced = True
R_smooth = 2.
ind_snap = 0; z_nbody = 1.
N_dim = 256 # particle mesh size; usually ppd
Lbox = 175.#Mpc/h
ppd = 256

# cosmological parameters
n_s = 0.96
Omega_c = 0.655
Omega_b = 0.045
h = 0.7
sigma8_m = 0.8

# create CCL cosmology object
cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

# scale the density as suggested in Modi et al.
D_z = ccl.growth_factor(cosmo,1./(1+z_nbody))

# sim_dir of simulation
if machine == 'alan':
    sim_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
    data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
    dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
elif machine == 'NERSC':
    sim_dir = "/global/cscratch1/sd/damonge/NbodySims/"+sim_name+"/"
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
    dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/"

# find all files
ic_fns = sorted(glob.glob(sim_dir+"ic_*"))
snap_fns = sorted(glob.glob(sim_dir+"snap_*"))
fof_fns = sorted(glob.glob(sim_dir+"fof_*.fits"))

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# return position of the particles and halos
lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)
#np.save(data_dir+"pos_ic.npy",lagr_pos)
np.save(data_dir+"pos_snap.npy",pos_snap)
np.save(data_dir+"pos_halo.npy",pos_halo)
print("Obtained positions and snapshots")
del pos_halo

# might want to split into two files here; or just do a for loop above

# load the 5 fields
ones, delta, delta_sq, nabla_sq, s_sq = load_fields(dens_dir,R_smooth,N_dim,Lbox,z_nbody)
print("Loaded all fields")

print(D_z)
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
