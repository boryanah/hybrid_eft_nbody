import numpy as np
import os
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt
import pyccl as ccl

from tools.power_spectrum import predict_Pk, predict_Pk_cross
from choose_parameters import load_dict
from scipy.interpolate import interp1d

# user choices
fit_type = 'power_hh'
#fit_type = 'power_both'
#fit_type = 'power_hm'
#fit_type = 'ratio'
#fit_type = 'ratio_both'
k_max = 0.5
k_min = 1.e-4
fit_shotnoise = False

# redshift choice
#z_nbody = 1.1
#z_s = np.array([0.,0.3,0.7,1.])
# TESTING CLASS
class_dir = "/home/boryanah/repos/AbacusSummit/Cosmologies/abacus_cosm000/"
#z_s = np.array([3.0, 2.5, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])[::-1]
#z_s = np.array([0.8, 0.5, 0.4, 0.3])[::-1]
#z_s = np.array([1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])[::-1]
z_s = np.array([1.1, 0.8, 0.5, 0.1])[::-1]
#z_s = np.array([1.4, 1.1, 0.8, 0.5])[::-1]
a_s = 1./(1+z_s)
# TESTING CLASS
z_nbody = 1.
# og
#z_nbody = z_s[0]

# name of the machine
machine = 'alan'
#machine = 'NERSC'

# simulation name
#sim_name = "AbacusSummit_hugebase_c000_ph000"
#sim_name = "AbacusSummit_hugebase_c000_ph000"
sim_name = "Sim256"

# load parameters
user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
R_smooth = user_dict['R_smooth']
data_dir = user_dict['data_dir']

# Cosmology
cosmo = ccl.Cosmology(**cosmo_dict)

# Redshift distributions
nz_s = np.exp(-((z_s-0.5)/0.05)**2/2)

# Bias
bz_s = 0.95/ccl.growth_factor(cosmo,a_s)

# This tracer will only include the density contribution
halos = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_s,nz_s), bias=(z_s,bz_s), mag_bias=None)

# change the order cause that's what CCL prefers
a_s = a_s[::-1]
bz_s = bz_s[::-1]

for i in range(len(a_s)):
    # which halo files are we loading
    z = z_s[i]
    a = a_s[i]

    # data directory
    # TESTING CLASS og is uncommented
    #data_dir = data_dir.replace('z1.000','z%.3f'%z)

    # load power spectra
    ks = np.load(data_dir+"ks.npy")
    # TESTING CLASS
    ks = ks[~np.isnan(ks)]
    print(data_dir)
    #Pk_hh = np.load(data_dir+"Pk_hh.npy")
    #Pk_mm = np.load(data_dir+"Pk_mm.npy")
    #Pk_hm = np.load(data_dir+"Pk_hm.npy")
        
    # apply cuts to the data
    k_cut = (ks < k_max) & (ks >= k_min)
    # TESTING CLASS
    #Pk_hh = Pk_hh[k_cut]
    #Pk_mm = Pk_mm[k_cut]
    #Pk_hm = Pk_hm[k_cut]
    ks = ks[k_cut]
    #ks[0] = k_min # without this line things go exponential
    print(ks[0])

    # TESTING CLASS
    print("abacus_cosm000.z%d_pk.dat"%(i+1),z)
    klin, Pklin = np.loadtxt(class_dir+"abacus_cosm000.z%d_pk.dat"%(i+1),unpack=True)
    print(klin.min(),klin.max(),ks.min(),ks.max())
    Pk_fun = interp1d(klin,Pklin)
    # TESTING helps next line helps to make things match
    ks = klin
    Pk_hh = bz_s[i]**2*Pk_fun(ks)
    
    # interpolate
    k_interp = np.logspace(np.log10(ks[0]),np.log10(ks[-1]),1000)
    Pk_fun = interp1d(ks,Pk_hh)
    log_pk = np.log(Pk_fun(k_interp))
    #log_pk = np.log(Pk_hh)
    
    # assemble things
    try:
        lpk_array = np.vstack((lpk_array,log_pk))
    except:
        lpk_array = log_pk


# testing
#z = np.linspace(0,1.2,1024)
#a_s = 1./(1+z)
#a_s = a_s[::-1]
#ks = np.logspace(-5,2,512)

# simple power spectrum for testing
lpk_array = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in a_s]))

# Create a Pk2D object
pk_tmp = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)

# wave numbers
ells = np.geomspace(2,1000,20)

# Compute power spectra with and without cutoff
cl_hh = ccl.angular_cl(cosmo, halos, halos, ells)
cl_hh_tmp = ccl.angular_cl(cosmo, halos, halos, ells, p_of_k_a=pk_tmp)


# Let's plot the result
plt.plot(ells, 1E4*cl_hh, 'r-', label='built-in tracer')
plt.plot(ells, 1E4*cl_hh_tmp, 'k--', label='custom tracer')
plt.xscale('log')
plt.xlabel('$\\ell$', fontsize=14)
plt.ylabel('$10^4\\times C_\\ell$', fontsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.savefig("figs/Cls.png")
#plt.ylim([0,0.03])
plt.show()
