

def load_dict(sim_name,machine):

    #machine = 'alan'
    machine = 'NERSC'

    # todo read stuff from sims
    R_smooth = 2.
    z_nbody = 1.
    ind_dict_gadget = {'1.000': 0, '0.500': 1}

    
    if 'Abacus' in sim_name:
        sim_code = 'abacus'
    else:
        sim_code = 'gadget'
    
    # data directory: abacus
    if sim_code == 'abacus':
        if machine == 'alan':
            data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/"
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/"
    elif sim_code == 'gadget':
        if machine == 'alan':
            sim_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
        elif machine == 'NERSC':
            sim_dir = "/global/cscratch1/sd/damonge/NbodySims/"+sim_name+"/"
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/"

    
    # cosmological parameters: abacus
    cosmo_dict_abacus = {'h': 0.6736,
                         'n_s': 0.9649,
                         'Omega_b': 0.02237/0.6736**2,
                         'Omega_c': 0.12/0.6736**2,
                         'sigma8': 0.807952}

    # cosmological parameters: gadget
    cosmo_dict_gadget = {'h': 0.7,
                         'n_s': 0.96,
                         'Omega_c': 0.655,
                         'Omega_b': 0.045,
                         'sigma8': 0.8}
    
    
    # user choices: abacus
    user_dict_abacus = {'sim_name': sim_name,
                        'interlaced': True,
                        'N_dim': 2304,
                        'ppd': 2304,
                        'z_ic': 49,
                        'z_nbody': 1.,
                        'Lbox': 2000.,
                        'n_chunks': 20,
                        'data_dir': data_dir,
                        'dens_dir': dens_dir,
                        'R_smooth': R_smooth}

    # user choices: gadget
    user_dict_gadget = {'sim_name': sim_name,
                        'interlaced': True,
                        'N_dim': 256,
                        'ppd': 256,
                        'z_ic': 49,
                        'ind_snap': ind_dict_gadget['%.3f'%(z_nbody)],
                        'z_nbody': z_nbody,
                        'Lbox': 175.,
                        'n_chunks': 1,
                        'data_dir': data_dir,
                        'dens_dir': dens_dir,
                        'sim_dir': sim_dir,
                        'R_smooth': R_smooth}

    
    if sim_code == 'abacus':
        return user_dict_abacus, cosmo_dict_abacus
    else:
        return user_dict_gadget, cosmo_dict_gadget
