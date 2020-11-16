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

# calculate chi2 (pretty evident)
def calculate_chi2(f_i):
    # require positivity of all parameters
    if np.sum(f_i < 0.) > 0:
        return np.inf
    
    # predict Pk for given bias params and apply the k-space cuts
    P_hh = predict_Pk(f_i,ks_all,Pk_all,k_lengths)
    P_hh = P_hh[k_cut]

    # predict Pk for given bias params and apply the k-space cuts
    P_hm = predict_Pk_cross(f_i,ks_all,Pk_all,k_lengths)
    P_hm = P_hm[k_cut]

    if fit_type == 'power':
        attempt = P_hh
        target = Pk_hh
    elif fit_type == 'power_both':
        attempt = np.hstack((P_hh,P_hm))
        target = Pk
    elif fit_type == 'ratio':
        attempt = P_hh/Pk_mm
        target = rat_hh
    elif fit_type == 'ratio_both':
        attempt = np.hstack((P_hh/Pk_mm,P_hm/Pk_mm))
        target = rat
    
    # compute chi2
    diff = attempt-target
    chi2 = np.dot(diff,np.dot(icov,diff))
    return chi2

# initial guess for bias parameters: F_i = {1,b_1,b_2,b_nabla,b_s}
b_1 = 1.2
b_2 = 0.4
b_nabla = 0.1
b_s = 0.2
x0 = np.array([b_1, b_2, b_nabla, b_s])
xtol = 1.e-6

# minimize using nelder-mead, powell
method = 'powell'
#method = 'nelder-mead'
res = minimize(calculate_chi2, x0, method=method,\
               options={'xtol': xtol, 'disp': True})
alpha = res.x
print(alpha)

print('alpha = ',alpha)
F_i = np.ones(5)
F_i[1:] = alpha

for i in range(5):
    for j in range(5):
        if i > j: continue
        print('F_%d F_%d = '%(i+1,j+1),F_i[i]*F_i[j])
        print("--------------------------")


# compute power spectrum for best-fit parameters
Pk_hh_best = predict_Pk(alpha,ks_all,Pk_all,k_lengths)
Pk_hh_best = Pk_hh_best[k_cut]
Pk_hm_best = predict_Pk_cross(alpha,ks_all,Pk_all,k_lengths)
Pk_hm_best = Pk_hm_best[k_cut]

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
