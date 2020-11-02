import numpy as np
import os
from nbodykit.lab import *
import pyccl as ccl
from scipy.optimize import minimize
import glob
import matplotlib.pyplot as plt
import time

from nbodykit.source.catalog import FITSCatalog

from tools.power_spectrum import get_Pk
from tools.read_abacus import read_abacus

def get_power():

    machine = 'alan'
    #machine = 'NERSC'

    #sim_code = 'abacus'
    sim_code = 'gadget'
    
    if sim_code == 'abacus':
        # user choices: abacus
        sim_name = "AbacusSummit_hugebase_c000_ph000"
        interlaced = True
        R_smooth = 2.
        N_dim = 2304 # particle mesh size; usually ppd
        ppd = 2304 # particle per dimension in the sim
        z_nbody = 1. # redshift where we measure power spectrum
        Lbox = 2000. # box size of the simulation [Mpc/h]
        n_chunks = 20

        # cosmological parameters: abacus
        h = 0.6736
        n_s = 0.9649
        Omega_b = 0.02237/h**2
        Omega_c = 0.12/h**2
        sigma8_m = 0.807952

    elif sim_code == 'gadget':
        # user choices: gadget
        sim_name = 'Sim256'
        interlaced = True
        R_smooth = 2.
        ind_snap = 0; z_nbody = 1.
        N_dim = 256 # particle mesh size; usually ppd
        Lbox = 175.#Mpc/h
        ppd = 256

        # cosmological parameters: gadget
        n_s = 0.96
        Omega_c = 0.655
        Omega_b = 0.045
        h = 0.7
        sigma8_m = 0.8

    if sim_code == 'abacus':
        if machine == 'alan':
            data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    elif sim_code == 'gadget':
        if machine == 'alan':
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
    
    
    # load simulation information; 
    pos_halo_fns = sorted(glob.glob(data_dir+"pos_halo_*"))
    pos_snap_fns = sorted(glob.glob(data_dir+"pos_ones_snap_*"))

    print("Obtained positions and snapshots")
    N_halo = np.sum(np.load(data_dir+"N_halo_lengths.npy"))
    
    # number density of halos and shot noise
    n_halo = N_halo/Lbox**3.
    P_sn = 1./n_halo

    # obtain the hh power spectrum
    ks, Pk_hh = get_Pk(pos_halo_fns,N_dim,Lbox,interlaced)
    print("Computed hh power spectrum")
    np.save(data_dir+"Pk_hh.npy",Pk_hh)
    np.save(data_dir+"Pk_hh-sn.npy",Pk_hh-P_sn)
    np.save(data_dir+"ks.npy",ks)

    # obtain the mm power spectrum
    ks, Pk_mm = get_Pk(pos_snap_fns,N_dim,Lbox,interlaced)
    print("Computed mm power spectrum")
    np.save(data_dir+"Pk_mm.npy",Pk_mm)

    # obtain the mm power spectrum
    ks, Pk_hm = get_Pk(pos_halo_fns,N_dim,Lbox,interlaced,pos2_fns=pos_snap_fns)
    print("Computed hm power spectrum")
    np.save(data_dir+"Pk_hm.npy",Pk_hm)


if __name__ == "__main__":

    t1 = time.time()
    get_power()
    t2 = time.time(); print("t = ",t2-t1)
