import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt

from nbodykit.lab import *

from choose_parameters import load_dict
from tools.compute_fields import get_fields, load_field, get_fields_bigfile, load_field_bigfile
from tools.read_gadget import get_density

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    # redshift choice (doesn't matter for the field calculation)
    z_nbody = 1.1
    
    #mode = 'convert_to_bigfile'
    #mode = 'resample'
    mode = ''
    
    want_bigfile = True
    testing = 0
    
    #machine = 'alan'
    machine = 'NERSC'

    sim_name = "AbacusSummit_hugebase_c000_ph000"
    #sim_name = 'Sim256'
    #sim_name = 'Sim1024'
    
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    dens_dir = user_dict['dens_dir']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    sim_code = user_dict['sim_code']
    sim_dir = user_dict['sim_dir']
    N_dim = user_dict['N_dim']
    Lbox = user_dict['Lbox']
    N_dim_new = int(N_dim//2)
    
    print("dens_dir = ",dens_dir)

    if mode == 'resample':
        print("Resampling is not recommended: best to regenerate ICs")
        # I have modified /anaconda3/envs/p3/lib/python3.6/site-packages/nbodykit/base/mesh.py
        mesh = BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
        # this is the version if unmodified
        #mesh = ArrayMesh(BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field').paint(mode='real', Nmesh=N_dim_new),BoxSize=Lbox)

        # save as a bigfile
        mesh.save_rigged(dens_dir+"density_%d.bigfile"%N_dim_new, mode='real', dataset='Field',Nmesh=N_dim_new)
    
    # for converting file to big file
    if mode == 'convert_to_bigfile':
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
    
    if want_bigfile:
        get_fields_bigfile(dens_dir,R_smooth,N_dim,Lbox)
    else:
        get_fields(dens_dir,R_smooth,N_dim,Lbox)
    print("obtained fields")
    

    if testing == False: return

    field_names = ["delta","delta_sq","nabla_sq","s_sq"]
    
    for field_name in field_names:
        if want_bigfile:
            field = load_field_bigfile(field_name,dens_dir,R_smooth)
        else:
            field = load_field(field_name,dens_dir,R_smooth)

        print(field.shape)
        plt.figure(1)
        plt.imshow(field[:,:,20])
        plt.colorbar()

        
if __name__ == "__main__":

    t1 = time.time()
    main()

    '''
    # doesn't actually work - perhaps different architecture
    p = Pool(10)    
    p.starmap(main, zip(repeat(1)))
    p.close()
    p.join()
    '''
    t2 = time.time(); print("t = ",t2-t1)


