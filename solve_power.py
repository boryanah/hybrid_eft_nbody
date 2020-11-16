import numpy as np
import os
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.power_spectrum import predict_Pk, predict_Pk_cross
from choose_parameters import load_dict

# user choices
fit_type = 'power'
#fit_type = 'power_both'
#fit_type = 'ratio'
#fit_type = 'ratio_both'
k_max = 0.3
k_min = 0.

machine = 'alan'
#machine = 'NERSC'

#sim_name = "AbacusSummit_hugebase_c000_ph000"
sim_name = "Sim256"

# load parameters
user_dict, cosmo_dict = load_dict(sim_name,machine)
R_smooth = user_dict['R_smooth']
data_dir = user_dict['data_dir']

# load power spectra
#Pk_hh = np.load(data_dir+"Pk_hh_mean.npy")
#Pk_hh = np.load(data_dir+"Pk_hh-sn.npy")
Pk_hh = np.load(data_dir+"Pk_hh.npy")
Pk_mm = np.load(data_dir+"Pk_mm.npy")
Pk_hm = np.load(data_dir+"Pk_hm.npy")
ks = np.load(data_dir+"ks.npy")

# apply cuts to the data
k_cut = (ks < k_max) & (ks >= k_min)
Pk_hh = Pk_hh[k_cut]
Pk_mm = Pk_mm[k_cut]
Pk_hm = Pk_hm[k_cut]
ks = ks[k_cut]

# load errorbars for plotting
Pk_hh_err = np.load(data_dir+"Pk_hh_err.npy")
Pk_hh_err = Pk_hh_err[k_cut]
Pk_hm_err = Pk_hm*0.3#np.load(data_dir+"Pk_hm_err.npy")
#Pk_hm_err = Pk_hm_err[k_cut]

# combine the ratios
# TODO has to be done properly with jackknifing
if fit_type == 'power':
    Pk_hh = Pk_hh
    Pk_hh_err = Pk_hh_err
    Pk_hh_err[0] = 1.e-6
    cov = np.diag(Pk_hh_err)
elif fit_type == 'power_both':
    Pk = np.hstack((Pk_hh,Pk_hm))
    Pk_err = np.hstack((Pk_hh_err,Pk_hm_err))
    Pk_err[0] = 1.e-6
    Pk_err[len(Pk_hh)] = 1.e-6
    cov = np.diag(Pk_err)
elif fit_type == 'ratio':
    rat_hh = Pk_hh/Pk_mm
    rat_hh_err = np.ones(len(rat_hh))*1.e-1
    rat_hh_err[0] = 1.e-6
    cov = np.diag(rat_hh_err)
elif fit_type == 'ratio_both':
    rat = np.hstack((Pk_hh/Pk_mm,Pk_hm/Pk_mm))
    rat_err = np.ones(len(rat))*1.e-1
    rat_err[0] = 1.e-6
    rat_err[len(Pk_hh)] = 1.e-6
    cov = np.diag(rat_err)
icov = np.linalg.inv(cov)

# load all 15 templates
ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)

# linear solution
Pk_all = Pk_all.reshape(int(len(ks_all)/k_lengths[0]),k_lengths[0])
Pk_all = Pk_all[:,k_cut]
P_hat = Pk_all.T
alpha = np.dot(np.linalg.inv(np.dot(np.dot(P_hat.T,icov),P_hat)),np.dot(np.dot(P_hat.T,icov),Pk_hh[:,None]))


print('alpha = ',alpha)
F_i = np.zeros(5)
# TESTING
F_i[0] = 1.#np.sqrt(alpha[0])
F_i[1] = (alpha[1])/F_i[0]
F_i[2] = (alpha[2])/F_i[0]
F_i[3] = (alpha[3])/F_i[0]
F_i[4] = (alpha[4])/F_i[0]

c = 0
for i in range(5):
    for j in range(5):
        if i > j: continue
        print('F_%d F_%d = '%(i+1,j+1),F_i[i]*F_i[j])
        print('alpha = ',alpha[c])
        c += 1
        print("--------------------------")

# compute power spectrum for best-fit
Pk_hh_best = np.dot(P_hat,alpha)
print(len(Pk_hh_best))
#Pk_best = Pk_best[k_cut]
Pk_hm_best = np.sum(np.array([P_hat[:,i]*F_i[i] for i in range(len(F_i))]),axis=0)
print(len(Pk_hm_best))
#Pk_hm_best = Pk_hm_best[k_cut]

# plot fit
plt.figure(1)
plt.errorbar(ks,Pk_hh,yerr=Pk_hh_err,color='black',label='halo-halo',zorder=1)
plt.plot(ks,Pk_hh_best,color='dodgerblue',label='halo-halo fit',zorder=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.legend()
plt.savefig("figs/Pk_hh_fit.png")

plt.figure(2)
plt.errorbar(ks,Pk_hm,yerr=Pk_hm_err,color='black',label='halo-matter',zorder=1)
plt.plot(ks,Pk_hm_best,color='dodgerblue',label='halo-matter fit',zorder=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.legend()
plt.savefig("figs/Pk_hm_fit.png")
plt.show()
