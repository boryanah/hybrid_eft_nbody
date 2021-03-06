import os
import glob

import numpy as np
import asdf

from tools.read_params import get_dict

def load_dict(z_nbody,sim_name,machine):
    # user choices
    #R_smooth = 4.
    #R_smooth = 2.
    #R_smooth = 1.
    R_smooth = 0 
    interlaced = True
    m_threshold = 1.e13
    
    if 'Abacus' in sim_name:
        sim_code = 'abacus'
    else:
        sim_code = 'gadget'

    # data directory: abacus
    if sim_code == 'abacus':
        if machine == 'alan':
            sim_dir = "/mnt/gosling2/bigsims/"+sim_name+"/halos/"
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
        elif machine == 'NERSC':
            #sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name+"/halos/"
            sim_dir = "/global/cscratch1/sd/boryanah/data_hybrid/tape_data/"+sim_name+"/halos/"
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            if 'c000_ph000' in sim_name:
                dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/AbacusSummit_hugebase_c000_ph000/"
                print("for any box that is cosmo 000 phase 000 can use the hugebase field files")
            else:
                dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/"

        cat_dir = os.path.join(sim_dir,"z%.3f"%z_nbody)
        n_chunks = len(glob.glob(os.path.join(cat_dir,'halo_info/halo_info_*.asdf')))

        f = asdf.open(os.path.join(cat_dir,'halo_info/halo_info_000.asdf'),lazy_load=True,copy_arrays=False)
        header = f['header']

        Lbox = header['BoxSize']
        ppd = header['ppd']
        m_part = header['ParticleMassHMsun']
        N_dim = 2304
        z_ic = 99
        
        # cosmology
        h = header['H0']/100.
        n_s = header['n_s']
        Omega_b = header['omega_b']/h**2
        Omega_c = header['omega_cdm']/h**2
        class_dict = get_dict(sim_name)
        sigma8 = np.float(class_dict['sigma8_m'])
        
    # data directory: gadget
    elif sim_code == 'gadget':
        if machine == 'alan':
            sim_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
        elif machine == 'NERSC':
            sim_dir = "/global/cscratch1/sd/damonge/NbodySims/"+sim_name+"/"
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/"

        ind_dict_gadget = {'1.000': 0, '0.700': 1, '0.300': 2, '0.000': 3}
        z_ic = 49

        # cosmology
        h = 0.7
        n_s = 0.96
        Omega_c = 0.26
        Omega_b = 0.04
        sigma8 = 0.8
        
        if sim_name == 'Sim256':
            N_dim = 256
            ppd = 256
            Lbox = 175.
            #n_chunks = 2
            n_chunks = 1
        elif sim_name == 'Sim1024':
            N_dim = 1024
            ppd = 1024
            Lbox = 700.
            n_chunks = 4

    # cosmological parameters
    cosmo_dict = {'h': h,
                  'n_s': n_s,
                  'Omega_c': Omega_c,
                  'Omega_b': Omega_b,
                  'sigma8': sigma8}
    
    # simulation
    user_dict = {'dk': 2.*np.pi/Lbox,
                 'ppd': ppd,
                 'z_ic': z_ic,
                 'Lbox': Lbox,
                 'N_dim': N_dim,
                 'z_nbody': z_nbody,
                 'sim_dir': sim_dir,
                 'n_chunks': n_chunks,
                 'data_dir': data_dir,
                 'dens_dir': dens_dir,
                 'R_smooth': R_smooth,
                 'sim_code': sim_code,
                 'sim_name': sim_name,
                 'interlaced': interlaced,
                 'mass_threshold': m_threshold}

    # special fields
    if sim_code == 'gadget':
        user_dict['ind_snap'] = ind_dict_gadget['%.3f'%(z_nbody)]
    elif sim_code == 'abacus':
        user_dict['m_part'] = m_part
    
    return user_dict, cosmo_dict
