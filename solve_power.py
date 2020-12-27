#!/usr/bin/env python3
'''
This is a script for solving analytically for the 5 bias parameters.

Usage: 
------ 
./solve_power.py --sim_name AbacusSummit_base_c000_ph000

'''

import glob
import os

import asdf
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
DEFAULTS['factor'] = 1.e-4
DEFAULTS['max_iter'] = 1000
DEFAULTS['tol'] = 1.e-3
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'
DEFAULTS['fit_type'] = 'power_hh' #'power_hm' #'power_both'

def solve(Pk_ij, Pk_hh_true, Pk_hm_true, icov, F, k_length, tol, factor, max_iter, fit_type):
    err = 1.e9
    iteration = 0
    F_old = np.ones((len(F),1))*1.e9
    while err > tol and iteration < max_iter:
        # fi fj Pkij and P_hat
        Pk_hh_guess, Pk_hm_guess, P_hat = get_P(Pk_ij, F, k_length)

        # subtract guess from the truth
        P_hh = Pk_hh_true - Pk_hh_guess
        P_hm = Pk_hm_true - Pk_hm_guess
        
        if fit_type == 'power_hh':
            P_h = P_hh
        elif fit_type == 'power_both':
            P_h = np.hstack((P_hh,P_hm)).T
            P_hat = np.vstack((P_hat,Pk_ij[0,1:,:].T))
        elif fit_type == 'power_hm':
            P_h = P_hm
            P_hat = Pk_ij[0,1:,:].T

        # solve matrix equation
        PTiCov = np.dot(P_hat.T,icov)
        iPTiCovP = np.linalg.inv(np.dot(PTiCov,P_hat))
        alpha = np.dot(iPTiCovP,np.dot(PTiCov,P_h[:,None]))

        # save new values
        F_old = F.copy()
        F[1:] += factor * alpha

        # compute error
        err = np.sqrt(np.sum(((F-F_old)/F)**2))

        # record iteration
        iteration += 1


    print("Fitting type = ",fit_type)
    print("Finished in %d iterations with bias values of "%iteration,F.T)
    return F


def get_P(Pk_ij, F_this, k_length):
    # guess for the halo-halo and halo-matter power spectrum
    Pk_hh_guess = np.einsum('i,ij...,j',F_this.flatten(), Pk_ij, F_this.flatten())
    Pk_hm_guess = np.einsum('i,i...',F_this.flatten(), Pk_ij[0,:,:])
    # computing P^i = \sum fj Pij
    P_hat = np.einsum('j,ij...',F_this.flatten(), Pk_ij)[:,1:]

    return Pk_hh_guess, Pk_hm_guess, P_hat 

def main(sim_name, sim_name_halo, z_nbody, z_ic, R_smooth, machine, fit_type, tol, factor, max_iter, fit_shotnoise=False):
    
    # power spectrum choices
    k_max = 0.5#0.3#0.5
    k_min = 0.#1.e-2#0
    
    # load parameters
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    R_smooth = user_dict['R_smooth']
    data_dir = user_dict['data_dir']
    Lbox = user_dict['Lbox']

    # which halo files are we loading
    halo_dir = data_dir.replace(sim_name,sim_name_halo)

    # load power spectra
    Pk_hh = np.load(halo_dir+"Pk_hh.npy")
    #Pk_mm = np.load(halo_dir+"Pk_mm.npy")
    # TESTING
    Pk_mm = np.load("data/AbacusSummit_base_c000_ph000/z1.100/r_smooth_0/Pk_mm.npy")
    Pk_hm = np.load(halo_dir+"Pk_hm.npy")
    ks = np.load(halo_dir+"ks.npy")

    # apply cuts to the data
    k_cut = (ks < k_max) & (ks >= k_min)
    Pk_hh = Pk_hh[k_cut]
    Pk_mm = Pk_mm[k_cut]
    Pk_hm = Pk_hm[k_cut]
    ks = ks[k_cut]
    dk = ks[1]-ks[0]
    
    # number of modes in each bin
    N_modes = ks**2*dk*Lbox**3/(2.*np.pi**2)
    # load errorbars for plotting
    Pk_hh_err = Pk_hh*np.sqrt(2./N_modes)
    Pk_hm_err = np.sqrt((Pk_hm**2+Pk_hh*Pk_mm)/N_modes)
    
    # combine the ratios
    # TODO has to be done properly with jackknifing
    Pk_hh_err[0] = 1.e6 #tuks
    cov_hh = np.diag(Pk_hh_err**2)

    Pk_both = np.hstack((Pk_hh,Pk_hm))
    Pk_both_err = np.hstack((Pk_hh_err,Pk_hm_err))
    Pk_both_err[len(Pk_hh)] = 1.e6
    cov_both = np.diag(Pk_both_err**2)

    Pk_hm_err[0] = 1.e6
    cov_hm = np.diag(Pk_hm_err**2)


    
    # load all 15 templates
    ks_all = np.load(data_dir+"ks_all.npy")
    Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth))) # og
    Pk_tmps = asdf.open(data_dir+"Pk_templates_0.asdf")['data'] # og
    #TESTING
    #data_dir = "data/AbacusSummit_base_c000_ph000/z1.100/r_smooth_1/"
    #Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(1))
    k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
    fields_tmp = ['1', 'b_1', 'b_2', 'b_{\\nabla^2}', 'b_s']
    
    # linear solution
    Pk_all = Pk_all.reshape(int(len(ks_all)/k_lengths[0]),k_lengths[0])
    Pk_all = Pk_all[:,k_cut]

    # shot noise params
    #Pk_sh = 1./n_bar # analytical shot noise
    F_size = 5

    Pk_ij = np.zeros((F_size,F_size,len(Pk_hh)))
    c = 0
    for i in range(F_size):
        for j in range(F_size):
            if i > j: continue
            
            # TESTING
            Pk_tmp = Pk_tmps[r'$('+fields_tmp[i]+','+fields_tmp[j]+r')$']
            Pk_tmp = np.interp(ks,Pk_tmps['ks'],Pk_tmp)
            # original
            #Pk_tmp = Pk_all[c]
            
            Pk_ij[i,j,:] = Pk_tmp
            if i != j: Pk_ij[j,i,:] = Pk_tmp
            c += 1


    # solution params
    F_start = np.ones((F_size,1)) # initial guess
    n_steps = 10

    # first solve varying all 4 parameters
    if fit_type == 'power_both':
        icov = np.linalg.inv(cov_both)
    elif fit_type == 'power_hh':
        icov = np.linalg.inv(cov_hh)
    elif fit_type == 'power_hm':
        icov = np.linalg.inv(cov_hm)
    F = solve(Pk_ij, Pk_hh, Pk_hm, icov, F_start, len(Pk_hh), tol, factor, max_iter, fit_type)
    #F = np.array([1.,-0.8277,-0.0424,-0.339,0.0355])
    #F = np.array([1, -1.01524924, 0.0075658, 0.0001073, -0.0052661])
    
    # compute power spectrum for best-fit
    Pk_hh_guess, Pk_hm_guess, P_hat = get_P(Pk_ij, F, len(Pk_hh))
    Pk_hh_best = Pk_hh_guess
    Pk_hm_best = Pk_hm_guess

    # compute the probability
    delta = Pk_hh_best-Pk_hh
    lnprob = np.einsum('i,ij,j',delta, np.linalg.inv(cov_hh), delta)
    lnprob *= -0.5 
    print("lnprob = ", lnprob)
    print("Pk_hh_truth = ", Pk_hh[::10])
    print("Pk_hh_best = ", Pk_hh_best[::10])
    print("Pk_hm_truth = ", Pk_hm[::10])
    print("Pk_hm_best = ", Pk_hm_best[::10])
    
    # plot solution
    plt.figure(1, figsize=(12,8))
    fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
    for i in range(len(F)):
        for j in range(len(F)):
            if i > j: continue
            label = r'$\langle '+fields[i]+","+fields[j]+r" \rangle$"
            Pk_tmp = Pk_ij[i,j,:]*F[i]*F[j]
            plt.plot(ks,Pk_tmp,ls='--',lw=2.,label=label)

    plt.errorbar(ks, Pk_hh, yerr=Pk_hh_err, color='black', label='halo-halo truth', zorder=1)
    plt.plot(ks, Pk_hh_best, color='dodgerblue', label='halo-halo fit', zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.legend()
    plt.savefig("figs/Pk_hh_fit.png")

    plt.figure(2)
    plt.errorbar(ks,Pk_hm,yerr=Pk_hm_err,color='black',label='halo-matter truth',zorder=1)
    plt.plot(ks,Pk_hm_best,color='dodgerblue',label='halo-matter fit',zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.legend()
    plt.savefig("figs/Pk_hm_fit.png")
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
    parser.add_argument('--tol', help='Tolerance for the solution', type=float, default=DEFAULTS['tol'])
    parser.add_argument('--max_iter', help='Maximum number of iterations', type=float, default=DEFAULTS['max_iter'])
    parser.add_argument('--factor', help='Size of the iterative steps', type=float, default=DEFAULTS['factor'])
    parser.add_argument('--fit_shotnoise', help='Fit for the shot noise (not working I think)', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
