#!/usr/bin/env python3
'''
This is a script for obtaining smooth templates that combine N-body simulations with velocileptors LPT code.

Usage:
------
./obtain_theory.py --z_nbody 0.8
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

# home directory
home = os.path.expanduser("~")
from choose_parameters import load_dict
from tools.read_params import get_dict

# colour table in HTML hex format
hexcols = ['#44AA99', '#117733', '#999933', '#88CCEE', '#332288', '#BBBBBB', '#4477AA',
           '#CC6677', '#AA4499', '#6699CC', '#AA4466', '#882255', '#661100', '#0099BB', '#DDCC77']

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_nbody'] = 1.1
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'

# function that computes linear derivatives: dPk/dp = (Pk_+ - Pk_-)/2h
def deriv_Pk(Pk_plus, Pk_minus, h):
    dPk = (Pk_plus-Pk_minus)/(2.*h)
    return dPk

# predict value given derivative, fiducial and difference: Pk = Pk_fid + dPk*delta
def predict_Pk(Pk_fid, dPk, delta):
    Pk_plus = Pk_fid + dPk*delta
    Pk_minus = Pk_fid - dPk*delta
    return Pk_plus, Pk_minus


def main(sim_name, z_nbody, R_smooth, machine):

    # parameter that is being tested
    #par_vary = 'omega_cdm' # missing 103
    par_vary = 'n_s'
    #par_vary = 'sigma8_cb'
    
    # now we need the choose parameters
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    data_dir = user_dict['data_dir']
    
    # load fiducial asdf file
    Pk_templates = asdf.open(os.path.join(data_dir, "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
    ks = Pk_templates['ks']
    print('ks = ', ks[:5], ks[-1])

    # get cosmological parameters for fiducial cosmology
    fid_dict = get_dict(sim_name)
    
    # available parameters
    # todo: cb or m
    pars = ['omega_b', 'omega_cdm', 'n_s', 'sigma8_cb']
    main_name = 'AbacusSummit_base'
    sim_par_dict = {'omega_b': [main_name+'_c100_ph000', main_name+'_c101_ph000'],
                   'omega_cdm': [main_name+'_c102_ph000', main_name+'_c103_ph000', main_name+'_c117_ph000', main_name+'_c118_ph000'],
                   'n_s': [main_name+'_c104_ph000', main_name+'_c105_ph000', main_name+'_c119_ph000', main_name+'_c120_ph000'],
                   'sigma8_cb': [main_name+'_c112_ph000', main_name+'_c113_ph000', main_name+'_c125_ph000', main_name+'_c126_ph000']}

    # simulations that vary par_vary
    sim_names = sim_par_dict[par_vary]

    # load templates for these simulations
    Pk_templates_large_plus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[0]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
    Pk_templates_large_minus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[1]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
    Pk_templates_small_plus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[2]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
    Pk_templates_small_minus = asdf.open(os.path.join(data_dir.replace(sim_name, sim_names[3]), "Pk_templates_%d.asdf"%(int(R_smooth))))['data']
    
    # get the relevant parameters for each simulation
    pars_dict = {}
    for sim in sim_names:
        dic = get_dict(sim)
        pars_dict[sim] = dic

    # derivative difference for the pair
    h_large = pars_dict[sim_names[0]][par_vary] - fid_dict[par_vary]
    h_small = pars_dict[sim_names[2]][par_vary] - fid_dict[par_vary]
    
    print(pars_dict[sim_names[0]][par_vary], fid_dict[par_vary])
    print("h's = ", h_large, h_small)
    # plot spectra
    for i, key in enumerate(Pk_templates.keys()):
        if key == 'ks': i -= 1; continue
        Pk_tmp = Pk_templates[key]
        Pk_tmp_plus = Pk_templates_large_plus[key]
        Pk_tmp_minus = Pk_templates_large_minus[key]
        dPk_tmp = deriv_Pk(Pk_tmp_plus, Pk_tmp_minus, h_large)
        Pk_pred_plus, Pk_pred_minus = predict_Pk(Pk_tmp, dPk_tmp, h_small)
        Pk_true_plus = Pk_templates_small_plus[key]
        Pk_true_minus = Pk_templates_small_minus[key]
        #print("deriv, pred/true = ", dPk_tmp, Pk_pred_plus/Pk_true_plus)
        
        #plt.subplot(2,3,plot_no)
        #plt.loglog(ks, Pk_true_plus, color=hexcols[i], label=key+' plus')
        #plt.loglog(ks, Pk_pred_plus, color=hexcols[i], ls='--')
        #plt.loglog(ks, Pk_true_minus, color=hexcols[15-i-1], label=key+' minus')
        #plt.loglog(ks, Pk_pred_minus, color=hexcols[15-i-1], ls='--')
        plt.plot(ks, np.ones(len(ks)), 'k--')
        plt.semilogx(ks, Pk_pred_plus/Pk_true_plus, color=hexcols[i], label=key+' plus')
        plt.semilogx(ks, Pk_pred_minus/Pk_true_minus, color=hexcols[15-i-1], label=key+' minus')
        plt.legend(ncol=1)

        plt.xlabel('k [h/Mpc]')
        #plt.ylabel(r'$P_{ab}$ [(Mpc/h)$^3$]')
        plt.ylabel(r'$P_{ab}^{\rm pred}/P_{ab}^{\rm true}$')
        plt.ylim([0.8, 1.2])
        plt.savefig("figs/deriv_"+par_vary+"_"+key+"_z%4.3f.png"%z_nbody)
        plt.close()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    args = parser.parse_args()
    args = vars(args)
    main(**args)
