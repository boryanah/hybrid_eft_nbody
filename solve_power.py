import numpy as np
import os
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.power_spectrum import predict_Pk

# user choices
interlaced = True
R_smooth = 2.
k_max = 0.4#1.
k_min = 0.#0.03 # todo: figure this out!!!!!!
z_nbody = 1.

simulation_code = 'gadget'
#simulation_code = 'abacus'

machine = 'alan'
#machine = 'NERSC'

#fit_type = 'ratio_both'
fit_type = 'power'
#fit_type = 'ratio'

if simulation_code == 'abacus':
    sim_name = "AbacusSummit_hugebase_c000_ph000"
    #small/AbacusSummit_small_c000_ph3046
    if machine == 'alan':
        data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    elif machine == 'NERSC':
        data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
elif simulation_code == 'gadget':
    sim_name = "Sim256"
    if machine == 'alan':
        data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
    elif machine == 'NERSC':
        data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody

# load power spectra
Pk_hh = np.load(data_dir+"Pk_hh_mean.npy")
#Pk_hh = np.load(data_dir+"Pk_hh-sn.npy")
#Pk_hh = np.load(data_dir+"Pk_hh.npy")
Pk_mm = np.load(data_dir+"Pk_mm.npy")
Pk_hm = np.load(data_dir+"Pk_hm.npy")
ks = np.load(data_dir+"ks.npy")

# apply cuts to the data
k_cut = (ks < k_max) & (ks >= k_min)
Pk_hh = Pk_hh[k_cut]
Pk_mm = Pk_mm[k_cut]
Pk_hm = Pk_hm[k_cut]
ks = ks[k_cut]

# combine the ratios
# TODO has to be done properly with jackknifing
if fit_type == 'ratio_both':
    rat_hh = np.hstack((Pk_hh/Pk_mm,Pk_hm/Pk_mm))
    rat_err = np.ones(len(rat_hh))*1.e-1
elif fit_type == 'ratio':
    rat_hh = Pk_hh/Pk_mm
    rat_err = np.ones(len(rat_hh))*1.e-1

# load errorbars for plotting
Pk_err = np.load(data_dir+"Pk_hh_err.npy")
Pk_err = Pk_err[k_cut]

# covariance matrix
if fit_type == 'ratio':
    cov = np.diag(rat_err)
else:
    cov = np.diag(Pk_err)
cov[0,0] = 1.
icov = np.linalg.inv(cov)
icov[0,0] = 1.e6

# load all 15 templates
ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)

# linear solution
Pk_all = Pk_all.reshape(int(len(ks_all)/k_lengths[0]),k_lengths[0])
Pk_all = Pk_all[:,k_cut]
P_hat = Pk_all.T
alpha = np.dot(np.linalg.inv(np.dot(np.dot(P_hat.T,icov),P_hat)),np.dot(np.dot(P_hat.T,icov),Pk_hh[:,None]))

# compute power spectrum for best-fit
Pk_best = np.dot(P_hat,alpha)
#Pk_best = Pk_best[k_cut]

# plot fit
plt.errorbar(ks,Pk_hh,yerr=Pk_err,color='black',label='halo-halo',zorder=1)
plt.plot(ks,Pk_best,color='dodgerblue',label='EFT-Hybrid',zorder=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.legend()
plt.savefig("figs/Pk_fit.png")
plt.show()
