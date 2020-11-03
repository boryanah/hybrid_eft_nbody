import numpy as np
import os
import pyccl as ccl
import glob
import matplotlib.pyplot as plt
import time

from nbodykit.lab import *
from nbodykit.utils import ScatterArray#can't broadcast

from tools.compute_fields import load_fields
from tools.power_spectrum import get_all_cross_ps, get_mesh
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
from load_dictionary import load_dict
import fitsio

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# save data
def save_pos(pos,weight,type_pos,data_dir):
    data = np.empty(pos.shape[0], dtype=[('Position', ('f8', 3)), ('Value', 'f8')])
    data['Position'] = pos
    data['Value'] = weight
    
    # write to a FITS file using fitsio
    fitsio.write(data_dir+"pos_"+type_pos+".fits", data, extname='Data')
    return


def get_templates():
    
    machine = 'alan'
    #machine = 'NERSC'

    sim_name = "AbacusSummit_hugebase_c000_ph000"
    sim_name = 'Sim256'

    user_dict, cosmo_dict = load_dict(sim_name,machine)
    interlaced = user_dict['interlaced']
    dens_dir = user_dict['dens_dir']
    data_dir = user_dict['data_dir']
    R_smooth = user_dict['R_smooth']
    n_chunks = user_dict['n_chunks']
    z_nbody = user_dict['z_nbody']
    N_dim = user_dict['N_dim']
    z_ic = user_dict['z_ic']
    Lbox = user_dict['Lbox']
    
    # names of the 5 fields
    field_names = ['ones', 'delta', 'delta_sq', 'nabla_sq', 's_sq']

    # load the cosmology
    cosmo = ccl.Cosmology(**cosmo_dict)

    # factor to scale the density as suggested in Modi et al.
    D_z_nbody = ccl.growth_factor(cosmo,1./(1+z_nbody))
    D_z_ic = ccl.growth_factor(cosmo,1./(1+z_nbody))
    D_growth = D_z_nbody/D_z_ic

    if rank == 0:
        # load the 5 fields
        fields, factors = load_fields(dens_dir,R_smooth,N_dim,Lbox)
        print("Loaded all fields")
    else:
        fields = None
        factors = None
    fields = comm.bcast(fields, root=0)
    factors = comm.bcast(factors, root=0)

    # create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop over all chunks
    N_halo_per_file = np.zeros(n_chunks,dtype=int)
    for i_chunk in range(n_chunks):
        if (rank-1) != i_chunk%size: continue
        
        if os.path.exists(data_dir+"pos_s_sq_snap_%03d.fits"%i_chunk):
            print("Data from chunk %d already saved, moving on"%i_chunk)
            continue
        
        if user_dict['sim_code'] == 'abacus':
            # load simulation information 
            lagr_pos, pos_snap, halo_table, header = read_abacus(sim_name,z_nbody,i_chunk)
            pos_halo = halo_table['x_L2com']

            # convert [-Lbox/2.,Lbox/2.] to [0,Lbox]
            pos_halo += Lbox/2.
            pos_snap += Lbox/2.
            lagr_pos += Lbox/2.
            
        elif user_dict['sim_code'] == 'gadget':
            # find all files, todo: fix for multiple chunks
            ic_fns = sorted(glob.glob(user_dict['sim_dir']+"ic_*"))
            snap_fns = sorted(glob.glob(user_dict['sim_dir']+"snap_*"))
            fof_fns = sorted(glob.glob(user_dict['sim_dir']+"fof_*.fits"))
            lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,user_dict['ind_snap'])

        save_pos(pos_halo,np.ones(pos_halo.shape[0]),"halo_%03d"%i_chunk,data_dir)
        N_halo_per_file[i_chunk] = pos_halo.shape[0]
        del pos_halo
            
        # get i, j, k for position on the density array
        lagr_ijk = (lagr_pos/(Lbox/N_dim)).astype(int)%N_dim
        del lagr_pos
        
        for key in fields.keys():
            if key == 'ones':
                weights = np.ones(pos_snap.shape[0])
            else:
                weights = (fields[key]*factors[key]*D_growth)[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
            #weights /= np.sum(weights)
            save_pos(pos_snap,weights,key+"_snap_%03d"%i_chunk,data_dir)
        del pos_snap, lagr_ijk

    if np.sum(N_halo_per_file) > 0:
        np.save(data_dir+"N_halo_lengths.npy",N_halo_per_file)
    print("Saved all particle and halo positions")
    del fields

    # wait until all data has been saved (might want to run the two separately after all) (or even split three-fold)
    while len(glob.glob(data_dir+"pos_s_sq_snap_*.fits")) < n_chunks: time.sleep(3)
    
    # get a mesh list for all 5 cases
    mesh_list = []
    for key in field_names:
        pos_snap_fns = sorted(glob.glob(data_dir+"pos_"+key+"_snap_*"))
        mesh = get_mesh(pos_snap_fns,N_dim,Lbox,interlaced)
        mesh_list.append(mesh)
    print("Obtained mesh lists for all fields")
    
    # compute all cross power spectra
    ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list)
    del mesh_list

    # save all power spectra
    np.save(data_dir+"ks_all.npy",ks_all)
    np.save(data_dir+"Pk_all_%d.npy"%(int(R_smooth)),Pk_all)
    np.save(data_dir+"k_lengths.npy",k_lengths)

    print("Saved all templates")

if __name__ == "__main__":

    t1 = time.time()
    get_templates()
    t2 = time.time(); print("t = ",t2-t1)
    
