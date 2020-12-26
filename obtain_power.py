#!/usr/bin/env python3
'''
This is a script for obtaining the halo-halo, halo-matter and matter-matter power spectra.

Usage: 
------ 
./obtain_power.py --z_nbody 0.8

'''

import os
import glob
import time
from memory_profiler import memory_usage

import numpy as np
import argparse
from nbodykit.lab import *
from nbodykit.source.catalog import FITSCatalog

from tools.power_spectrum import get_Pk
from choose_parameters import load_dict

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_nbody'] = 1.1
DEFAULTS['z_ic'] = 99.
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'

def main(sim_name, z_nbody, z_ic, R_smooth, machine):
    # which power spectra to compute
    #compute_pks = ['Pk_hh', 'Pk_hm', 'Pk_mm']
    compute_pks = ['Pk_hm', 'Pk_mm']
    
    # load dictionary
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    interlaced = user_dict['interlaced']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    n_chunks = user_dict['n_chunks']
    N_dim = user_dict['N_dim']
    Lbox = user_dict['Lbox']
    dk = user_dict['dk']
    m_threshold = user_dict['mass_threshold']
    
    # load simulation information; 
    pos_halo_fns = sorted(glob.glob(data_dir+"pos_halo_*"))
    pos_snap_fns = sorted(glob.glob(data_dir+"pos_ones_snap_*"))    

    # obtain the hh power spectrum
    if 'Pk_hh' in compute_pks:
        ks, Pk_hh = get_Pk(pos_halo_fns,N_dim,Lbox,interlaced,dk=dk,m_thr=m_threshold)
        print("Computed hh power spectrum")
        np.save(data_dir+"Pk_hh.npy",Pk_hh)
        np.save(data_dir+"ks.npy",ks)

    # obtain the mm power spectrum
    if 'Pk_mm' in compute_pks:
        ks, Pk_mm = get_Pk(pos_snap_fns,N_dim,Lbox,interlaced,dk=dk)
        print("Computed mm power spectrum")
        np.save(data_dir+"Pk_mm.npy",Pk_mm)

    # obtain the mm power spectrum
    if 'Pk_hm' in compute_pks:
        ks, Pk_hm = get_Pk(pos_halo_fns,N_dim,Lbox,interlaced,pos2_fns=pos_snap_fns,dk=dk)
        print("Computed hm power spectrum")
        np.save(data_dir+"Pk_hm.npy",Pk_hm)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--z_ic', help='N-body initial redshift', type=float, default=DEFAULTS['z_ic'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    args = parser.parse_args()
    args = vars(args)
    main(**args)
    #mem_usage = memory_usage(main(),interval=1., timeout=None)
    #print('Maximum memory usage: %s MB' % np.max(mem_usage))

