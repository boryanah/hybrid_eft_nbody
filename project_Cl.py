#!/usr/bin/env python3
'''
This is a script for testing the projection of 3D power spectra onto 2D.

Usage:
------
./project_Cl.py --z_nbody 0.8
'''

import os
import glob

import numpy as np
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyccl as ccl

from tools.power_spectrum import predict_Pk, predict_Pk_cross
from choose_parameters import load_dict

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_nbody'] = 1.1
DEFAULTS['z_ic'] = 99.
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'

def project_Cl(cosmo, tracer, Pk_a_s, ks, a_s, want_plot=False):
    # number of redshifts
    num_zs = Pk_a_s.shape[0]
    
    # change the order cause that's what CCL prefers
    i_sort = np.argsort(a_s)
    a_s = a_s[i_sort]
    assert num_zs == len(a_s), "Different number of input spectra and redshifts"

    # TESTING
    Pk_a_s = np.abs(Pk_a_s)
    
    # take the logarithm
    lpk_array = np.log(Pk_a_s)
    
    # simple power spectrum for testing
    #lpk_array = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in a_s]))
    #cl_tt = ccl.angular_cl(cosmo, tracer, tracer, ells)
    
    # Create a Pk2D object
    pk_tmp = ccl.Pk2D(a_arr=a_s, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)

    # wave numbers
    ells = np.geomspace(2,1000,20)

    # Compute power spectra with and without cutoff
    cl_tt_tmp = ccl.angular_cl(cosmo, tracer, tracer, ells, p_of_k_a=pk_tmp)
    
    '''
    # generating fake data
    Cl_err = cl_tt*np.sqrt(2./(2*ells+1.))
    cov = np.diag(Cl_err**2)
    np.save("data_power/cl_gg.npy",cl_tt)
    np.save("data_power/ells.npy",ells)
    np.save("data_power/cov_cl_gg.npy",cov)
    '''

    if want_plot:
        # Let's plot the result
        plt.plot(ells, 1E4*cl_tt, 'r-', label='built-in tracer')
        plt.plot(ells, 1E4*cl_tt_tmp, 'k--', label='custom tracer')
        plt.xscale('log')
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylabel('$10^4\\times C_\\ell$', fontsize=14)
        plt.legend(loc='upper right', fontsize=12, frameon=False)
        plt.savefig("figs/Cls.png")
        plt.show()

    return ells, cl_tt_tmp

def main(sim_name, z_nbody, z_ic, R_smooth, machine, want_plot=False):
    # user choices
    k_max = 0.5
    k_min = 1.e-4

    # redshift choice
    #z_s = np.array([3.0, 2.5, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])
    z_s = np.array([1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])
    z_s = np.sort(z_s)
    a_s = 1./(1+z_s)

    # test with class
    home = os.path.expanduser("~")
    class_dir = home+"/repos/AbacusSummit/Cosmologies/abacus_cosm000/"

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
    galaxies = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_s, nz_s), bias=(z_s, bz_s), mag_bias=None)

    # read in CLASS power spectra
    ks, Pk = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(0+1),unpack=True)
    Pk_a_s = np.zeros((len(a_s),len(ks)))
    for i in range(len(a_s)):
        print(i)
        Pk_a_s[i,:] = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(i+1))[:,1]

    
    # generating fake data
    k = np.load("data/AbacusSummit_base_c000_ph006/z1.100/ks.npy")
    dk = k[1]-k[0]
    Lbox = 2000.
    N_modes = k**2*dk*Lbox**3./(2.*np.pi**2)
    for i in range(len(a_s)):
        # ccl expects Mpc units todo: improve and ask
        h = cosmo_dict['h']
        Pk_gg = ccl.nonlin_matter_power(cosmo, k*h, a_s[i])*h**3
        cov = np.diag(Pk_gg**2*(2./N_modes))
        np.save("data_power/pk_gg_z%4.3f.npy"%z_s[i],Pk_gg)
        np.save("data_power/ks.npy",k)
        np.save("data_power/cov_pk_gg_z%4.3f.npy"%z_s[i],cov)
    
    # load k and P(k,a)
    ells, cl_tt_tmp = project_Cl(cosmo, galaxies, Pk_a_s, ks, a_s, k_min, k_max, want_plot)
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--z_ic', help='N-body initial redshift', type=float, default=DEFAULTS['z_ic'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    parser.add_argument('--want_plot', help='Plot the final result', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
