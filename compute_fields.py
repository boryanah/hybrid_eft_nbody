import numpy as np
import pyccl as ccl
from util import get_fields
from nbodykit.lab import *
import os

from compaso_halo_catalog import CompaSOHaloCatalog

# Simulation used: AbacusSummit_hugebase_c000_ph000, 27 times lower resolution than rest of sims.
# For more specs, see https://abacussummit.readthedocs.io/en/latest/simulations.html

# parameter choices for simulation
ppd = 2304 # particle per dimension in the sim
z_ic = 49. # redshift of initial conditions
z_nbody = 1. # redshift where we measure power spectrum
Lbox = 2000. # box size of the simulation [Mpc/h]
cpd = 455 # cells per dimension (internal to code)

# power spectrum parameters
interlaced = True

# simulation name
sim_name = "AbacusSummit_hugebase_c000_ph000"

# directory where the halo and particle catalogs are saved
cat_dir = "/mnt/store/AbacusSummit/"+sim_name+"/halos/"

# cosmological parameters
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

# load the cosmology
cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)
D_z = ccl.growth_factor(cosmo,1./(1+z_nbody))

# user choices
N_dim = ppd # particle mesh size


###############################################
####         Compute the IC fields         ####
###############################################

# load the raw binary file (slightly slower than np.load)
#fileName = "/mnt/alan1/boryanah/ICs/density2304"
#density_ic = np.fromfile(fileName, dtype=np.float32).reshape(ppd,ppd,ppd)
# load the density field (49 GB, 2304^3 cells)
density_ic = np.load("/mnt/alan1/boryanah/ICs/density.npy")

# scale the density as suggested in Modi et al.
density_scaled = D_z*density_ic

# the fields are i = {1,delta,delta^2,nabla^2 delta,s^2} 
ones = np.ones(density_scaled.shape)
delta = density_scaled
delta_sq = delta**2
nabla_sq, s_sq = get_fields(delta, Lbox, N_dim, fields=["nabla_sq","s_sq"])

# subtract mean
nabla_sq -= np.mean(nabla_sq)
s_sq -= np.mean(s_sq)

# compute weights with made-up bias parameters: F_i = {1,b_1,b_2,b_nabla,b_s}
b_1 = 1.2
b_2 = 0.4
b_nabla = 0.1
b_s = 0.2

# compute weighted field for this choice of bias parameters
weight_field = ones + b_1*delta + b_2*delta_sq + b_s*s_sq + b_nabla*nabla_sq

###############################################
####         Load N-body simulation        ####
###############################################

# load particle subsamples
# catalog directory
catdir = os.path.join(cat_dir,"z%.3f"%z_nbody)

# load halo catalog and 10% particle subsample
cat = CompaSOHaloCatalog(catdir, load_subsamples='AB_all', fields=['N'], unpack_bits = True)

# halo catalog -- not needed
#halo_table = cat.halos[halo_ind_lc]
#header = cat.header
#N_halos = len(cat.halos)
#print("N_halos = ",N_halos)

# load the pid, position and lagrangian positions
#pid = cat.subsamples['pid']
pos_mat = cat.subsamples['pos']
lagr_pos = cat.subsamples['lagr_pos']
lagr_ijk = ((lagr_pos/Lbox+0.5)*N_dim).astype(int)%N_dim

# compute the weights for each particle
weights = weight_field[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]


###############################################
####         Compute Power Spectra         ####
###############################################

# calculate power spectrum of the particles
matter = {}
matter['Position'] = pos_mat
matter['Weight'] = weights

cat = ArrayCatalog(matter)
mesh_mat = cat.to_mesh(window='tsc',Nmesh=Nmesh,BoxSize=BoxSize)

r_mat = FFTPower(mesh_mat, mode='1d')
ks = r_mat.power['k']
P_mat = r_mat.power['power']

# calculate power spectrum of the galaxies
# TODO: load one of your HOD mock catalogs (eh, disc broke down)
galaxies = {}
galaxies['Position'] = pos_gal

cat = ArrayCatalog(galaxies)
mesh_gal = cat.to_mesh(window='tsc',Nmesh=Nmesh,BoxSize=BoxSize,interlaced=interlaced,compensated=False)
compensation = mesh.CompensateTSC
mesh = mesh.apply(compensation, kind='circular', mode='complex')

r_gal = FFTPower(mesh_gal, mode='1d')
ks = r_gal.power['k']
P_gal = r_gal.power['power']#.real

# can also compute cross power spectra
#r_mat_gal = FFTPower(first=mesh_gal, second=mesh_mat, mode='1d')#, dk=0.005, kmin=0.01)   
#ks = r_mat_gal.power['k'] # [kpc/h]^-1
#P_mat_gal = r_mat_gal.power['power']

