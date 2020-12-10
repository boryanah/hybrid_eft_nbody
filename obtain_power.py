import numpy as np
import os
from nbodykit.lab import *
import glob
import time
from memory_profiler import memory_usage

from nbodykit.source.catalog import FITSCatalog

from tools.power_spectrum import get_Pk
from choose_parameters import load_dict

def main():
    # redshift choice
    z_nbody = 1.1
    #zs = [1.,0.7,0.3,0.]
    #z_nbody = zs[3]
    
    compute_pks = ['Pk_hh', 'Pk_hm', 'Pk_mm']
    #compute_pks = ['Pk_hh']

    #machine = 'alan'
    machine = 'NERSC'

    sim_name = 'AbacusSummit_hugebase_c000_ph000'
    #sim_name = 'AbacusSummit_hugebase_c000_ph001'
    #sim_name = 'Sim256'
    
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

if __name__ == "__main__":

    t1 = time.time()
    main()
    #mem_usage = memory_usage(main(),interval=1., timeout=None)
    #print('Maximum memory usage: %s MB' % np.max(mem_usage))
    t2 = time.time(); print("t = ",t2-t1)
