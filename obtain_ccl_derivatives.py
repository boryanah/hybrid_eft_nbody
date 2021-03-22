#!/usr/bin/env python3
'''
This is a script for obtaining smooth templates that combine N-body simulations with velocileptors LPT code.

Usage:
------
./obtain_derivatives.py --help
'''

import time
import sys
import os

import asdf
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# home directory
home = os.path.expanduser("~")
from choose_parameters import load_dict
from tools.read_params import get_dict
from obtain_theory import save_asdf
import pyccl as ccl

# colour table in HTML hex format
hexcols = ['#44AA99', '#117733', '#999933', '#88CCEE', '#332288', '#BBBBBB', '#4477AA',
           '#CC6677', '#AA4499', '#6699CC', '#AA4466', '#882255', '#661100', '#0099BB', '#DDCC77']

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['z_templates'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1]
DEFAULTS['R_smooth'] = 0.
#DEFAULTS['machine'] = 'NERSC'
DEFAULTS['machine'] = 'alan'
DEFAULTS['pars_vary'] = ['omega_b', 'omega_cdm', 'n_s', 'sigma8_cb']


def search(new_cosmo, theta_def, h_ini=0.55, h_fin=0.85, prec=1.e3, tol_t=1.e-3):    
    # array of hubble parameter values to search through
    hs = np.arange(h_ini*prec,h_fin*prec)/prec
    N_h = len(hs)
    
    # allowed tolerance b/n the new theta and the def
    this_theta = new_cosmo.theta_s_100()
    print("New 100*theta_s = ",this_theta)

    iterations = 0
    left = 0 # Determines the starting index of the list we have to search in
    right = N_h-1 # Determines the last index of the list we have to search in
    mid = (right + left)//2

    while(np.abs(this_theta-theta_def)>tol_t): # If this is not our search element
        # If the current middle element is less than x then move the left next to mid
        # Else we move right next to mid
        new_cosmo.set({'h': hs[mid]}) # .6736
        new_cosmo.compute()
        print("iter, h = ",iterations,", ",hs[mid])
        this_theta = new_cosmo.theta_s_100()
        if  this_theta < theta_def:
            left = mid + 1
        else:
            right = mid - 1
        mid = (right + left)//2
        iterations += 1
        print('Delta(100*theta_s) = ',np.abs(this_theta-theta_def))
        if right-left == 1: break

    # Final output
    print('iterations = ',str(iterations))
    print('h = ',new_cosmo.h())
    print('100*theta_s = ',new_cosmo.theta_s_100())
    print('Delta(100*theta_s) = ',np.abs(new_cosmo.theta_s_100()-theta_def))
    return new_cosmo.h()


def key2str(key):
    forbidden = ['$','{','(',',','\\','^2',')','}']
    string = key
    for s in forbidden:
        if ',' == s:
            string = string.replace(s, '_')
        else:
            string = string.replace(s, '')

    if 'nabla' in key:
        string = string.replace('nabla', 'n')

    if 'b_' in key:
        string = string.replace('b_', 'b')
        
    return string

# function that computes linear derivatives: dPk/dp = (Pk_+ - Pk_-)/2h
def deriv_Pk(Pk_plus, Pk_minus, Pk_mid, h, method='both'):
    if method == 'both':
        dPk = (Pk_plus-Pk_minus)/(2.*h)
    elif method == 'right':
        dPk = (Pk_plus-Pk_mid)/(h)
    elif method == 'left':
        dPk = (Pk_mid-Pk_minus)/(h)
    return dPk

# predict value given derivative, fiducial and difference: Pk = Pk_fid + dPk*delta
def predict_Pk(Pk_fid, dPk, delta):
    Pk_plus = Pk_fid + dPk*delta
    Pk_minus = Pk_fid - dPk*delta
    return Pk_plus, Pk_minus


def main(sim_name, z_templates, R_smooth, machine, pars_vary, check_derivatives=False):

    # convert the templates to floats and sort since angular cl expects that
    z_templates = np.array([float(z) for z in z_templates])
    z_templates = np.sort(z_templates)[::-1]

    fid_Pk_dPk_templates = {}
    for k, z_nbody in enumerate(z_templates):
        # get the name of that
        z_str = 'ztmp%d'%k
        
        # now we need the choose parameters
        if machine == 'alan':
            #data_dir =  home+"/repos/hybrid_eft_nbody/data/%s/z%4.3f/r_smooth_%d/"%(sim_name,z_nbody,int(R_smooth))
            data_dir =  home+"/repos/hybrid_eft_nbody/data/%s/z%4.3f/"%(sim_name,z_nbody)
        else:
            user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
            data_dir = user_dict['data_dir']

        # load fiducial asdf file
        Pk_templates = asdf.open(os.path.join(data_dir, "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
        ks = Pk_templates['ks']

        # get cosmological parameters for fiducial cosmology
        fid_dict = get_dict(sim_name)

        # available parameters
        main_name = 'AbacusSummit_base'
        sim_par_dict = {'omega_b': [main_name+'_c100_ph000', main_name+'_c101_ph000', main_name+'_c100_ph000', main_name+'_c101_ph000'],
                       'omega_cdm': [main_name+'_c102_ph000', main_name+'_c103_ph000', main_name+'_c117_ph000', main_name+'_c118_ph000'],
                       'n_s': [main_name+'_c104_ph000', main_name+'_c105_ph000', main_name+'_c119_ph000', main_name+'_c120_ph000'],
                       'sigma8_cb': [main_name+'_c112_ph000', main_name+'_c113_ph000', main_name+'_c125_ph000', main_name+'_c126_ph000']}

        for j, par_vary in enumerate(pars_vary):
            # simulations that vary par_vary
            sim_names = sim_par_dict[par_vary]

            # load templates for these simulations
            # temporarily let's use the fine derivatives for z = 1.1, 0.8, and 0.5 for the omega_cdm simulations
            if (par_vary == 'omega_cdm') and (z_nbody in [0.5, 0.8, 1.1]):
                sim_names[0] = sim_names[2]
                sim_names[1] = sim_names[3]
            elif (z_nbody in [0.4, 0.3, 0.2, 0.1]) or (par_vary == 'omega_b'):
                sim_names[2] = sim_names[0]
                sim_names[3] = sim_names[1]
                
            Pk_templates_large_plus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[0]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
            Pk_templates_large_minus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[1]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
            if check_derivatives:
                Pk_templates_small_plus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[2]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
                Pk_templates_small_minus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[3]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']

            # get the relevant parameters for each simulation
            pars_dict = {}
            for sim in sim_names:
                dic = get_dict(sim)
                pars_dict[sim] = dic

            # derivative difference for the pair
            h_large = pars_dict[sim_names[0]][par_vary] - fid_dict[par_vary]
            if check_derivatives:
                h_small = pars_dict[sim_names[2]][par_vary] - fid_dict[par_vary]
                

            # filling a dictionary of halofit values for derivatives and fiducial
            Pk_halofit = {}
            names_hf = ['large_plus', 'large_minus', 'small_plus', 'small_minus']
            for s, name_hf in enumerate(names_hf):
                h = pars_dict[sim_names[s]]['h']
                omega_cdm = pars_dict[sim_names[s]]['omega_cdm']
                omega_b = pars_dict[sim_names[s]]['omega_b']
                n_s = pars_dict[sim_names[s]]['n_s']
                A_s = pars_dict[sim_names[s]]['A_s']
                param_dict = {'transfer_function': 'boltzmann_class'}
                param_dict['h'] = float(h)
                param_dict['Omega_c'] = float(omega_cdm/h**2)
                param_dict['Omega_b'] = float(omega_b/h**2)
                param_dict['n_s'] = float(n_s)
                param_dict['A_s'] = float(A_s)
                cosmo_ccl = ccl.Cosmology(**param_dict)
                Pk_halofit[name_hf] = ccl.nonlin_matter_power(cosmo_ccl, ks*h, a=1./(1+z_nbody))*h**3
            cosmo_ccl = ccl.Cosmology(A_s=float(fid_dict['A_s']), n_s=float(fid_dict['n_s']), Omega_b=float(fid_dict['omega_b'])/float(fid_dict['h'])**2, Omega_c=float(fid_dict['omega_cdm'])/float(fid_dict['h'])**2, h=float(fid_dict['h']), transfer_function='boltzmann_class')
            Pk_halofit['fiducial'] = ccl.nonlin_matter_power(cosmo_ccl, Pk_templates['ks']*h, a=1./(1+z_nbody))*fid_dict['h']**3


            '''
            # TESTING ratio halofit to templates
            print("z = ", z_nbody)
            ratio = Pk_templates[r'$(1,1)$']/Pk_halofit['fiducial']
            np.save("Pk11_Pkhf.npy", ratio)
            np.save("ks.npy", ks)
            plt.plot(ks, ratio, label=r'$z = %.1f$'%z_nbody)
            plt.legend()
            plt.xscale('log')
            plt.xlabel(r"$k \ [h/{\rm Mpc}]$")
            plt.ylabel(r"$P_{11}(k)/P_{\rm hf}(k)$")
            plt.savefig("Pk11_Pkhf.png")
            plt.show()
            '''
            
            plot_no = 1
            plt.subplots(3, 5, figsize=(18,10))
            # plot spectra and save derivatives
            for i, key in enumerate(Pk_templates.keys()):

                
                # save the fiducial power spectrum
                if key == 'ks': i -= 1; continue
                rat_Pk_tmp = Pk_templates[key]/Pk_templates[r'$(1,1)$']#Pk_halofit['fiducial']
                rat_Pk_tmp_plus = Pk_templates_large_plus[key]/Pk_templates_large_plus[r'$(1,1)$']#Pk_halofit['large_plus']
                rat_Pk_tmp_minus = Pk_templates_large_minus[key]/Pk_templates_large_minus[r'$(1,1)$']#Pk_halofit['large_minus']
                drat_Pk_tmp = deriv_Pk(rat_Pk_tmp_plus, rat_Pk_tmp_minus, rat_Pk_tmp, h_large, method='both')
                
                # need to save the derivatives
                fid_Pk_dPk_templates[z_str+'_'+key2str(key)+'_'+par_vary] = drat_Pk_tmp
                if j == 0:
                    fid_Pk_dPk_templates[z_str+'_'+key2str(key)] = rat_Pk_tmp

                
                if check_derivatives:
                    Pk_pred_plus, Pk_pred_minus = predict_Pk(rat_Pk_tmp, drat_Pk_tmp, h_small)
                    Pk_pred_plus *= Pk_halofit['small_plus']
                    Pk_pred_minus *= Pk_halofit['small_minus']
                    Pk_true_plus = Pk_templates_small_plus[key]
                    Pk_true_minus = Pk_templates_small_minus[key]
                    
                    print(key2str(key))
                    if key2str(key) == '1_1':
                        print(par_vary, h_small, z_nbody)
                        np.save("Pk_11_"+par_vary+"_ztmp%d.npy"%k, Pk_pred_plus)
                    
                    plt.subplot(3,5,plot_no)
                    #plt.loglog(ks, Pk_true_plus, color=hexcols[i], label=key+' plus')
                    #plt.loglog(ks, Pk_pred_plus, color=hexcols[i], ls='--')
                    #plt.loglog(ks, Pk_true_minus, color=hexcols[15-i-1], label=key+' minus')
                    #plt.loglog(ks, Pk_pred_minus, color=hexcols[15-i-1], ls='--')
                    plt.plot(ks, np.ones(len(ks)), 'k--')
                    plt.semilogx(ks, Pk_pred_plus/Pk_true_plus, color=hexcols[i], label=key)#+' plus')
                    plt.semilogx(ks, Pk_pred_minus/Pk_true_minus, color=hexcols[15-i-1], ls='--')#, label=key+' minus')
                    plt.legend(ncol=1)

                    if plot_no >= (3-1)*5+1:
                        plt.xlabel('k [h/Mpc]')
                    #plt.ylabel(r'$P_{ab}$ [(Mpc/h)$^3$]')
                    if plot_no % 5 == 1:
                        plt.ylabel(r'$P_{ab}^{\rm pred}/P_{ab}^{\rm true}$')
                    plt.ylim([0.9, 1.1])
                    plot_no += 1
            plt.savefig("figs/deriv_"+par_vary+"__z%4.3f.png"%z_nbody)
            plt.close()
            
    # add the wavenumbers
    fid_Pk_dPk_templates['ks'] = ks
    
    # add the header with fiducial cosmological parameters and the redshifts of the templates [how do elegantly]
    #header = {par: fid_dict[par] for par in pars_vary}
    #header['h'] = fid_dict['h']
    header = fid_dict.copy()
    header.pop(r'notes')
    header.pop(r'root')
    header.pop(r'sigma8_cb') # og
    header.pop(r'sigma8_m')
    header.pop(r'w0_fld')
    header.pop(r'wa_fld')

    print(fid_Pk_dPk_templates.keys())
    # todo check that you can call class get the theta star value and then output the H0 value

    from classy import Class
    target_param_dict = header.copy()

    target_cosmo = Class()
    target_cosmo.set(target_param_dict)
    target_cosmo.compute()
    theta_target = target_cosmo.theta_s_100()
    print("Target 100*theta_s = ",theta_target)
    print('h = ', target_param_dict['h'])

    #target_cosmo.set({'omega_cdm': 0.11})
    target_cosmo.compute()
    new_cosmo = target_cosmo

    # this_cosmo can have same params as target_cosmo changing only the 4 parameters that are being varied and g
    h = search(new_cosmo, theta_target)
    header['theta_s_100'] = theta_target
    header['sigma8_cb'] = fid_dict['sigma8_cb']
    header['w0_fld'] = fid_dict['w0_fld']
    header['wa_fld'] = fid_dict['wa_fld']

    # I think we need to solve for h at each iteration? in ccl_class given theta star
    for i in range(len(z_templates)):
        header['ztmp%d'%i] = z_templates[i]
        
    #print(header.keys())
    print(header.items())
        
    # save as asdf file
    save_asdf(fid_Pk_dPk_templates,"fid_rat_Pk_dPk_templates_%d.asdf"%(int(R_smooth)), data_dir, header=header)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_templates', nargs='+', help='Redshifts of the templates', default=DEFAULTS['z_templates'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    parser.add_argument('--pars_vary', nargs='+', help='Parameters we are testing', default=DEFAULTS['pars_vary'])
    parser.add_argument('--check_derivatives', help='Check derivatives against truth', action='store_true')
    
    args = parser.parse_args()
    args = vars(args)
    main(**args)
