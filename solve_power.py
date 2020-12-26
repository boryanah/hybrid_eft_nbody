#!/usr/bin/env python3
'''
This is a script for solving analytically for the 5 bias parameters.

Usage: 
------ 
./solve_power.py --sim_name AbacusSummit_base_c000_ph000

'''

import glob
import os

import numpy as np
import argparse
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.power_spectrum import predict_Pk, predict_Pk_cross
from choose_parameters import load_dict

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['sim_name_halo'] = DEFAULTS['sim_name']
DEFAULTS['z_nbody'] = 1.1
DEFAULTS['z_ic'] = 99.
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'
DEFAULTS['fit_type'] = 'power_hh' #'power_hm' #'power_both'


def solve(Pk_hh_true, Pk_hm_true, cov, F_ini, k_length, tol, max_iter, fit_type, fix_f0):
    err = 1.e9
    iteration = 0
    f_shot = 0.
    F_old = np.ones((len(F_ini),1))*1.e9
    F = F_ini
    icov = np.linalg.inv(cov)
    while err > tol and iteration < max_iter:
        # P_hat
        P_guess, P_hat = get_P(F,k_length,fix_f0)
        P_hh = Pk_hh_true - P_guess
        if fit_shotnoise:
            P_sh = Pk_sh - f_shot*Pk_const
            P_hh = P_hh_true - Pk_sh

        # subtract guess from the truth
        P_hm = Pk_hm_true - np.dot(F.T,Pk_ij[0,:,:]).flatten()
        
        if fit_type == 'power_hh':
            P_h = P_hh
            if fit_shotnoise:
                P_hat = np.vstack((P_hat.T,Pk_const)).T
        elif fit_type == 'power_both':
            P_h = np.hstack((P_hh,P_hm)).T

            # if setting F[0] = 1
            if fix_f0:
                P_hat = np.vstack((P_hat,F[0]*Pk_ij[0,1:,:].T))
            else:
                P_hat = np.vstack((P_hat,F[0]*Pk_ij[0,:,:].T))
        elif fit_type == 'power_hm':
            P_h = P_hm

            # if setting F[0] = 1
            if fix_f0:
                P_hat = F[0]*Pk_ij[0,1:,:].T
            else:
                P_hat = F[0]*Pk_ij[0,:,:].T

        # solve matrix equation
        PTiCov = np.dot(P_hat.T,icov)
        iPTiCovP = np.linalg.inv(np.dot(PTiCov,P_hat))
        alpha = np.dot(iPTiCovP,np.dot(PTiCov,P_h[:,None]))

        # save new values
        F_old = F.copy()

        # if setting F[0] = 1
        if fix_f0:
            # alpha has 4 dimensions that need to be added to F
            F[1:] += 0.1 * alpha#[:len(F_old)-1]
        else:
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
    return F, f_shot


def get_P(F_this,k_length,fix_f0):
    P_guess = np.zeros(k_length)
    
    # if setting F[0] = 1
    if fix_f0:
        # we decrease the dimension since excluding F0
        P_hat = np.zeros((k_length,len(F_this)-1))
    else:
        P_hat = np.zeros((k_length,len(F_this)))

    for i in range(len(F_this)):
        P_hat_i = np.zeros(k_length)
        for j in range(len(F_this)):
            # template cross correlating i and j fields
            P_ij = Pk_ij[i,j,:]
            # bias parameter fi, fj
            f_i = F_this[i]
            f_j = F_this[j]

            # guess for the halo-halo power spectrum
            P_guess += f_i*f_j*P_ij
            # computing P^i = \sum fj Pij
            P_hat_i += f_j*P_ij

        # if setting F[0] = 1
        if fix_f0:
            # correct the indexing
            if i == 0: continue
            P_hat[:,i-1] = 2.*P_hat_i
        else:
            P_hat[:,i] = 2.*P_hat_i 
            
    return P_guess, P_hat 

def main(sim_name, sim_name_halo, z_nbody, z_ic, R_smooth, machine, fit_type, fit_shotnoise=False):

    # power spectrum choices
    k_max = 0.5
    k_min = 0.
    
    # load parameters
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    R_smooth = user_dict['R_smooth']
    data_dir = user_dict['data_dir']

    # which halo files are we loading
    halo_dir = data_dir.replace(sim_name,sim_name_halo)

    # load power spectra
    Pk_hh = np.load(halo_dir+"Pk_hh.npy")
    Pk_mm = np.load(halo_dir+"Pk_mm.npy")
    Pk_hm = np.load(halo_dir+"Pk_hm.npy")
    ks = np.load(halo_dir+"ks.npy")
    N_modes = len(ks)

    # apply cuts to the data
    k_cut = (ks < k_max) & (ks >= k_min)
    Pk_hh = Pk_hh[k_cut]
    Pk_mm = Pk_mm[k_cut]
    Pk_hm = Pk_hm[k_cut]
    ks = ks[k_cut]

    print("err/true = ",np.sqrt(2./N_modes))
    # load errorbars for plotting
    Pk_hh_err = Pk_hh*np.sqrt(2./N_modes)
    Pk_hm_err = np.sqrt((Pk_hm**2+Pk_hh*Pk_mm)/N_modes)

    # combine the ratios
    # TODO has to be done properly with jackknifing
    Pk_hh_err[0] = 1.e-6    
    cov_hh = np.diag(Pk_hh_err)

    Pk_both = np.hstack((Pk_hh,Pk_hm))
    Pk_both_err = np.hstack((Pk_hh_err,Pk_hm_err))
    Pk_both_err[len(Pk_hh)] = 1.e-6
    cov_both = np.diag(Pk_both_err)

    Pk_hm_err[0] = 1.e-6
    cov_hm = np.diag(Pk_hm_err)

    # load all 15 templates
    ks_all = np.load(data_dir+"ks_all.npy")
    Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
    k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)

    # linear solution
    Pk_all = Pk_all.reshape(int(len(ks_all)/k_lengths[0]),k_lengths[0])
    Pk_all = Pk_all[:,k_cut]
    #P_hat = Pk_all.T


    # shot noise params
    #Pk_sh = 1./n_bar # analytical shot noise
    Pk_sh = 0. #  [note that we have already subtracted it from Pk_hh]
    Pk_const = np.ones(len(Pk_hh))
    F_size = 5

    Pk_ij = np.zeros((F_size,F_size,len(Pk_hh)))
    c = 0
    for i in range(F_size):
        for j in range(F_size):
            if i > j: continue
            Pk_ij[i,j,:] = Pk_all[c]
            if i != j: Pk_ij[j,i,:] = Pk_all[c]
            c += 1


    # solution params
    tol = 1.e-3 # choose tolerance
    F_start = np.ones((F_size,1)) # initial guess
    n_steps = 10
    max_iter = 1000

    # first solve varying all 5 parameters
    fix_F0 = True# False
    fit_type = 'power_hh'
    F, f_shot = solve(Pk_hh, Pk_hm, cov_hh, F_start, len(Pk_hh), tol, max_iter, fit_type, fix_F0)
    '''
    # next fix F0 and change it slowly until it gets to 1 and a stable solution is found
    fix_F0 = True
    #fit_type = 'power_hh'
    fit_type = 'power_both'
    print(F[0][0])
    step_size = (F[0][0]-1.)/n_steps
    for i in range(n_steps):
        F[0][0] -= step_size
        print(F)
        #F, f_shot = solve(Pk_hh, Pk_hm, cov_hh, F, len(Pk_hh), tol, max_iter, fit_type, fix_F0)
        F, f_shot = solve(Pk_hh, Pk_hm, cov_both, F, len(Pk_hh), tol, max_iter, fit_type, fix_F0)
    '''
    # compute power spectrum for best-fit
    P_guess, P_hat = get_P(F,len(Pk_hh),fix_F0)
    Pk_hh_best = P_guess
    #Pk_best = Pk_best[k_cut]
    Pk_hm_best = np.dot(F.T,Pk_ij[0,:,:]).flatten()
    #P_hat[:,0]
    #Pk_hm_best = Pk_hm_best[k_cut]

    # compute the probability
    delta = Pk_hh_best-Pk_hh
    lnprob = np.einsum('i,ij,j',delta, np.linalg.inv(cov_hh), delta)
    lnprob *= -0.5 
    print("lnprob = ", lnprob)


    # plot fit
    plt.figure(1,figsize=(12,8))
    fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
    for i in range(len(F)):
        for j in range(len(F)):
            if i > j: continue
            label = r'$\langle '+fields[i]+","+fields[j]+r" \rangle$"
            Pk_tmp = Pk_ij[i,j,:]*F[i]*F[j]
            plt.plot(ks,Pk_tmp,ls='--',lw=1.,label=label)

    plt.errorbar(ks,Pk_hh,yerr=Pk_hh_err,color='black',label='halo-halo',zorder=1)
    plt.plot(ks,Pk_hh_best,color='dodgerblue',label='halo-halo fit',zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.legend()
    plt.savefig("figs/Pk_hh_fit.pdf")

    plt.figure(2)
    plt.errorbar(ks,Pk_hm,yerr=Pk_hm_err,color='black',label='halo-matter',zorder=1)
    plt.plot(ks,Pk_hm_best,color='dodgerblue',label='halo-matter fit',zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.legend()
    plt.savefig("figs/Pk_hm_fit.pdf")
    plt.show()


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name for the templates', default=DEFAULTS['sim_name'])
    parser.add_argument('--sim_name_halo', help='Simulation name for the halo power spectra', default=DEFAULTS['sim_name_halo'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--z_ic', help='N-body initial redshift', type=float, default=DEFAULTS['z_ic'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    parser.add_argument('--fit_type', help='Would you like to fit the auto, the cross or both power spectra', default=DEFAULTS['fit_type'])
    parser.add_argument('--fit_shotnoise', help='Fit for the shot noise (not working I think)', action='store_true'))
    args = parser.parse_args()
    args = vars(args)
    main(**args)
