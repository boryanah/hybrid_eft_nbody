import numpy as np
import os
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt

from tools.power_spectrum import predict_Pk, predict_Pk_cross
from choose_parameters import load_dict

# user choices
fit_type = 'power_hh'
#fit_type = 'power_both'
#fit_type = 'power_hm'
#fit_type = 'ratio'
#fit_type = 'ratio_both'
k_max = 0.3
k_min = 0.
fit_shotnoise = False

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
if fit_type == 'power_hh':
    Pk_hh = Pk_hh
    Pk_hh_err = Pk_hh_err
    Pk_hh_err[0] = 1.e-6

    # TESTING
    #Pk_hh_err = np.hstack((Pk_hh_err,np.ones(len(Pk_hh))*1.e-6))
    
    cov = np.diag(Pk_hh_err)
elif fit_type == 'power_both':
    Pk = np.hstack((Pk_hh,Pk_hm))
    Pk_err = np.hstack((Pk_hh_err,Pk_hm_err))
    Pk_err[len(Pk_hh)] = 1.e-6
    cov = np.diag(Pk_err)
elif fit_type == 'power_hm':
    Pk = Pk_hm
    Pk_err = Pk_hm_err
    Pk_err[0] = 1.e-6
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
#P_hat = Pk_all.T
        
def get_P(F_this,k_length):
    P_guess = np.zeros(k_length)
    P_hat = np.zeros((k_length,len(F_this)))
    for i in range(len(F_this)):
        P_hat_i = np.zeros(k_length)
        for j in range(len(F_this)):
            P_ij = Pk_ij[i,j,:]
            f_i = F_this[i]
            f_j = F_this[j]

            P_guess += f_i*f_j*P_ij
            P_hat_i += f_j*P_ij

        P_hat[:,i] = P_hat_i
    return P_guess, P_hat 

# initial guess
F = np.ones((5,1))
F_old = np.ones((5,1))*1.e9

# choose tolerance
tol = 1.e-3
k_length = len(Pk_hh)
err = 1.e9
iteration = 0

# shot noise params
#Pk_sh = 1./n_bar # ideal shot noise
Pk_sh = 0. #  [note that we have already subtracted it from Pk_hh]
f_shot = 0. # initial value
Pk_const = np.ones(k_length)

Pk_ij = np.zeros((len(F),len(F),k_length))
c = 0
for i in range(len(F)):
    for j in range(len(F)):
        if i > j: continue
        Pk_ij[i,j,:] = Pk_all[c]
        if i != j: Pk_ij[j,i,:] = Pk_all[c]
        c += 1

while err > tol:
    P_guess, P_hat = get_P(F,k_length)
    P_hh = Pk_hh - P_guess
    if fit_shotnoise:
        P_sh = Pk_sh - f_shot*Pk_const
        P_hh = P_hh - Pk_sh
    P_hm = Pk_hm - P_hat[:,0]

    if fit_type == 'power_hh':
        P_h = P_hh
        if fit_shotnoise:
            P_hat = np.vstack((P_hat.T,Pk_const)).T
    elif fit_type == 'power_both':
        P_h = np.hstack((P_hh,P_hm)).T
        P_hat = np.vstack((P_hat,F[0]*Pk_ij[0,:,:].T))
    elif fit_type == 'power_hm':
        P_h = P_hm
        P_hat = F[0]*Pk_ij[0,:,:].T

    # solve matrix equation
    PTiCov = np.dot(P_hat.T,icov)
    iPTiCovP = np.linalg.inv(np.dot(PTiCov,P_hat))
    alpha = np.dot(iPTiCovP,np.dot(PTiCov,P_h[:,None]))

    # save new values
    F_old = F.copy()
    F += 0.1 * alpha[:len(F_old)]
    err = np.sum(((F-F_old)/F)**2)
    
    if fit_shotnoise:
        P_hh -= Pk_sh-f_shot*Pk_const
        f_shot_old = f_shot
        f_shot += 0.1*alpha[-1]
        err += ((f_shot-f_shot_old))**2

    # compute error
    err = np.sqrt(err)

    # record iteration
    iteration += 1

print("Fitting type = ",fit_type)
print("Finished in %d iterations with bias values of "%iteration,F.T,f_shot)

# compute power spectrum for best-fit
P_guess, P_hat = get_P(F,k_length)
Pk_hh_best = P_guess
#Pk_best = Pk_best[k_cut]
Pk_hm_best = P_hat[:,0]
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
