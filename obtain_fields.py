#!/usr/bin/env python3
'''
This is a script for obtaining the fields and saving them as bigfile files. The code also allows converting the native density field into a bigfile.

Usage: 
------ 
./obtain_fields.py --sim_name AbacusSummit_base_c000_ph000

'''

import time
import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import argparse
from nbodykit.lab import *

from choose_parameters import load_dict
from tools.compute_fields import get_fields, load_field, get_fields_bigfile, load_field_bigfile
from tools.read_gadget import get_density

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

def main(sim_name, z_nbody, z_ic, R_smooth, machine, testing=False, convert_to_bigfile=False, resample=False):    

    # TESTING THIS IS SINCE WE ONLY USE ALAN FOR CONVERSION
    if machine == 'alan':
        print("I am gonna assume you are converting an AbacusSummit density file [PPD=2304, L=2000] into bigfile")
        # alternative if working from alan with the linear derivative fields
        N_dim = 2304
        Lbox = 2000.
        dens_dir = "/mnt/store1/boryanah/"+sim_name+"_ICs/"
        sim_code = 'abacus'
    else:
        # load dictionary
        user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
        dens_dir = user_dict['dens_dir']
        data_dir = user_dict['data_dir']
        R_smooth = user_dict['R_smooth']
        sim_code = user_dict['sim_code']
        sim_dir = user_dict['sim_dir']
        N_dim = user_dict['N_dim']
        Lbox = user_dict['Lbox']
        N_dim_new = int(N_dim//2)


    # print directory of the density field
    print("dens_dir = ",dens_dir)
    
    # for converting file to big file
    if convert_to_bigfile:
        try:
            mesh = np.load(dens_dir+"density_%d.npy"%N_dim)
        except:
            if sim_code == 'abacus':
                density = np.fromfile(dens_dir+"density%d"%N_dim, dtype=np.float32).reshape(N_dim,N_dim,N_dim)
            elif sim_code == 'gadget':
                density = get_density(sim_dir,Lbox,N_dim)
        mesh = ArrayMesh(density, BoxSize=Lbox)
        mesh.save(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
        print("converted into a bigfile")
        return

    if resample:
        print("Warning: Resampling is not recommended. Best to regenerate ICs")
        # I have modified /anaconda3/envs/p3/lib/python3.6/site-packages/nbodykit/base/mesh.py
        mesh = BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
        # this is the version if unmodified
        #mesh = ArrayMesh(BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field').paint(mode='real', Nmesh=N_dim_new),BoxSize=Lbox)

        # save as a bigfile
        mesh.save_rigged(dens_dir+"density_%d.bigfile"%N_dim_new, mode='real', dataset='Field',Nmesh=N_dim_new)
        return
        
    # compute fields
    get_fields_bigfile(dens_dir,R_smooth,N_dim,Lbox)
    print("obtained fields")
    
    # display fields
    if testing:
        field_names = ["delta","delta_sq","nabla_sq","s_sq"]

        for field_name in field_names:
            
            field = load_field_bigfile(field_name,dens_dir,R_smooth)
            print(field.shape)
            plt.figure(1)
            plt.imshow(field[:,:,20])
            plt.colorbar()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--z_ic', help='N-body initial redshift', type=float, default=DEFAULTS['z_ic'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    parser.add_argument('--testing', help='Test the fields by displaying them', action='store_true')
    parser.add_argument('--convert_to_bigfile', help='Convert the density field into a bigfile file', action='store_true')
    parser.add_argument('--resample', help='Resample the density field shrinking each dimension by a factor of 2', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)

    '''
    # doesn't actually work - perhaps different architecture
    p = Pool(10)    
    p.starmap(main, zip(repeat(1)))
    p.close()
    p.join()
    '''


