import numpy as np
import os
import pyccl as ccl
import glob
import time
import matplotlib.pyplot as plt

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray

#from multiprocessing import Pool
#from itertools import repeat

from tools.power_spectrum import resample_mesh
from load_dictionary import load_dict
from tools.compute_fields import get_fields, load_field, get_fields_bigfile, load_field_bigfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def calculate_fields():

    mode = 'convert_to_bigfile'
    #mode = 'resample'
    
    want_bigfile = True
    testing = 1
    
    machine = 'alan'
    #machine = 'NERSC'

    sim_name = "AbacusSummit_hugebase_c000_ph000"
    #sim_name = 'Sim256'

    user_dict, cosmo_dict = load_dict(sim_name,machine)
    dens_dir = user_dict['dens_dir']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    N_dim = user_dict['N_dim']
    Lbox = user_dict['Lbox']
    N_dim_new = int(N_dim//2)

    
    print("dens_dir = ",dens_dir)

    if mode == 'resample':
        # best to regenerate ICs
        # B.H. NOTE THAT I HAVE CHANGED THE NBODYKIT FILE: /anaconda3/envs/p3/lib/python3.6/site-packages/nbodykit/base/mesh.py
        mesh = BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
        #mesh = ArrayMesh(BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field').paint(mode='real', Nmesh=N_dim_new),BoxSize=Lbox)
        mesh.save_rigged(dens_dir+"density_%d.bigfile"%N_dim_new, mode='real', dataset='Field',Nmesh=N_dim_new)
                    
    
    if mode == 'convert_to_bigfile':
        # for converting file to big file
        mesh = ArrayMesh(np.load(dens_dir+"density_%d.npy"%N_dim), BoxSize=Lbox)
        mesh.save(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
        print("converted into a bigfile")

    
    if want_bigfile:
        get_fields_bigfile(dens_dir,R_smooth,N_dim,Lbox)
    else:
        get_fields(dens_dir,R_smooth,N_dim,Lbox)
    print("obtained fields")
    

    if testing == False: return
    minus_field_name = "mnabla_sq"

    #field_names = ["delta","delta_sq","nabla_sq","s_sq"]
    field_names = ["nabla_sq"]
    for field_name in field_names:
        field = load_field_bigfile(minus_field_name,dens_dir,R_smooth)
        #field = np.load(dens_dir+field_name+"_%d.npy"%R_smooth)
        if want_bigfile:
            field_test = load_field_bigfile(field_name,dens_dir,R_smooth)
        else:
            field_test = load_field(field_name,dens_dir,R_smooth)

        print(field_test.shape)
        plt.figure(1)
        plt.imshow(field[:,:,20])
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(field_test[:,:,20])
        plt.colorbar()
        plt.show()

        
if __name__ == "__main__":

    t1 = time.time()
    calculate_fields()

    '''
    # doesn't actually work - perhaps different architecture
    p = Pool(10)    
    p.starmap(calculate_fields, zip(repeat(1)))
    p.close()
    p.join()
    '''
    t2 = time.time(); print("t = ",t2-t1)


