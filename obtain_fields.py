import numpy as np
import os
import pyccl as ccl
import glob
import time

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray

from tools.power_spectrum import resample_mesh
from load_dictionary import load_dict
from tools.compute_fields import get_fields, load_field, get_fields_bigfile, load_field_bigfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def calculate_fields():

    want_bigfile = True
    
    testing = 0
    want_resample = False
    convert_to_bigfile = False
    
    machine = 'alan'
    #machine = 'NERSC'

    #sim_name = "AbacusSummit_hugebase_c000_ph000"
    sim_name = 'Sim256'

    user_dict, cosmo_dict = load_dict(sim_name,machine)
    dens_dir = user_dict['dens_dir']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    N_dim = user_dict['N_dim']
    Lbox = user_dict['Lbox']
    N_dim_new = int(N_dim//2)

    
    print("dens_dir = ",dens_dir)

    if convert_to_bigfile:
        # for converting file to big file
        if want_resample:
            mesh = ArrayMesh(ArrayMesh(np.load(dens_dir+"density.npy"), BoxSize=Lbox).paint(mode='real', Nmesh=N_dim_new))
            mesh.save(dens_dir+"density_%d.bigfile"%N_dim_new, mode='real', dataset='Field')
        else:
            mesh = ArrayMesh(np.load(dens_dir+"density.npy"), BoxSize=Lbox)
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
        field = np.load(dens_dir+field_name+"_%d.npy"%R_smooth)
        if want_bigfile:
            field_test = load_field_bigfile(field_name,dens_dir,R_smooth)
        else:
            field_test = load_field(field_name,dens_dir,R_smooth)

        print(field_test.shape)
        plt.figure(1)
        plt.imshow(field[:,:,20])

        plt.figure(2)
        plt.imshow(field_test[:,:,20])
        plt.show()

        
if __name__ == "__main__":

    t1 = time.time()
    calculate_fields()
    t2 = time.time(); print("t = ",t2-t1)


