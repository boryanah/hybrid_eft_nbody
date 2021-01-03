#!/usr/bin/env python3
'''
This is a script for obtaining the 15 templates coresponding to the cross-power spectra between all 5 fields.

Usage: 
------ 
./obtain_templates.py --z_nbody 0.8

'''

import os
import glob
import time
from memory_profiler import memory_usage

import numpy as np
import argparse
from nbodykit.lab import *

from tools.power_spectrum import get_all_cross_ps, get_mesh
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
from choose_parameters import load_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

DEFAULTS = {}
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_nbody'] = 1.1
DEFAULTS['z_ic'] = 99.
DEFAULTS['R_smooth'] = 0.
DEFAULTS['machine'] = 'NERSC'

def main(sim_name, z_nbody, z_ic, R_smooth, machine):
    # load dictionary
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    interlaced = user_dict['interlaced']
    dens_dir = user_dict['dens_dir']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    n_chunks = user_dict['n_chunks']
    z_nbody = user_dict['z_nbody']
    N_dim = user_dict['N_dim']
    z_ic = user_dict['z_ic']
    Lbox = user_dict['Lbox']
    dk = user_dict['dk']

    field_names = ['ones', 'delta', 'delta_sq', 'nabla_sq', 's_sq']

    # get a mesh list for all 5 cases
    mesh_list = []
    for key in field_names:
        if key == 'ones':
            pos_snap_fns = sorted(glob.glob(data_dir+"pos_delta_snap_*"))
        else:
            pos_snap_fns = sorted(glob.glob(data_dir+"pos_"+key+"_snap_*"))
        mesh = get_mesh(key, pos_snap_fns, N_dim, Lbox, interlaced)
        mesh_list.append(mesh)
    print("Obtained mesh lists for all fields")
    
    # compute all cross power spectra
    ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list,dk=dk)
    del mesh_list
    print("Computed cross power spectra of all fields")
    
    # save all power spectra
    np.save(data_dir+"ks_all.npy",ks_all)
    np.save(data_dir+"Pk_all_%d.npy"%(int(R_smooth)),Pk_all)
    np.save(data_dir+"k_lengths.npy",k_lengths)
    print("Saved all templates")

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
