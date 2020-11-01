import numpy as np
import os
from nbodykit.lab import *
import pyccl as ccl
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.power_spectrum import get_Pk
from tools.read_abacus import read_abacus

machine = 'alan'
#machine = 'NERSC'

# user choices, see https://abacussummit.readthedocs.io/en/latest/simulations.html
sim_name = "AbacusSummit_hugebase_c000_ph000"
interlaced = True
N_dim = 2304
Lbox = 2000. # box size of the simulation [Mpc/h]
z_nbody = 1. 

if machine == 'alan':
    data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
elif machine == 'NERSC':
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    
# load simulation information; todo: chunks and figure out parallel computation
pos_halo = np.load(data_dir+"pos_halo.npy")
pos_snap = np.load(data_dir+"pos_snap.npy")
N_halo = pos_halo.shape[0]
print("Obtained positions and snapshots")

# number density of halos and shot noise
n_halo = N_halo/Lbox**3.
P_sn = 1./n_halo

# obtain the hh power spectrum
ks, Pk_hh = get_Pk(pos_halo,N_dim,Lbox,interlaced)
Pk_hh = Pk_hh.astype(np.float64)
print("Computed hh power spectrum")
np.save(data_dir+"Pk_hh.npy",Pk_hh)
np.save(data_dir+"Pk_hh-sn.npy",Pk_hh-P_sn)
np.save(data_dir+"ks.npy",ks)

# obtain the mm power spectrum
ks, Pk_mm = get_Pk(pos_snap,N_dim,Lbox,interlaced)
Pk_mm = Pk_mm.astype(np.float64)
print("Computed mm power spectrum")
np.save(data_dir+"Pk_mm.npy",Pk_mm)

# obtain the mm power spectrum
ks, Pk_hm = get_Pk(pos_halo,N_dim,Lbox,interlaced,pos2=pos_snap)
Pk_hm = Pk_hm.astype(np.float64)
print("Computed hm power spectrum")
np.save(data_dir+"Pk_hm.npy",Pk_hm)
