import numpy as np
import os
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.power_spectrum import predict_Pk
from choose_parameters import load_dict

# user choices
#fit_type = 'ratio_both'
fit_type = 'power'
#fit_type = 'ratio'
k_max = .3#.6 works
k_min = 0.#0.03 

machine = 'alan'
#machine = 'NERSC'

#sim_name = "AbacusSummit_hugebase_c000_ph000"
sim_name = "Sim256"

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

def calculate_chi2(f_i):
    # require positivity
    if np.sum(f_i < 0.) > 0:
        return np.inf
    
    # predict Pk for given bias params
    Pk = predict_Pk(f_i,ks_all,Pk_all,k_lengths)

    # apply the k-space cuts
    Pk = Pk[k_cut]
    
    if fit_type == 'ratio':
        this = Pk/Pk_mm
        target = rat_hh
    elif fit_type == 'power':
        this = Pk
        target = Pk_hh
    elif fit_type == 'ratio_both':
        print("not implemented"); quit()
    
    # compute chi2
    dPk = this-target
    chi2 = np.dot(dPk,np.dot(icov,dPk))
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
F_i = np.zeros(5)
F_i[0] = 1.#np.sqrt(alpha[0])
F_i[1] = (alpha[0])/F_i[0]#(alpha[1])/F_i[0]
F_i[2] = (alpha[1])/F_i[0]#(alpha[2])/F_i[0]
F_i[3] = (alpha[2])/F_i[0]#(alpha[3])/F_i[0]
F_i[4] = (alpha[3])/F_i[0]#(alpha[4])/F_i[0]

c = 0
for i in range(5):
    for j in range(5):
        if i > j: continue
        print('F_%d F_%d = '%(i+1,j+1),F_i[i]*F_i[j])
        #print('alpha = ',alpha[c])
        c += 1
        print("--------------------------")


# compute power spectrum for best-fit parameters
Pk_best = predict_Pk(alpha,ks_all,Pk_all,k_lengths)
Pk_best = Pk_best[k_cut]

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
