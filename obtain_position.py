import numpy as np
import os
import pyccl as ccl
import glob
import time

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray # can't broadcast, only parts

from tools.compute_fields import load_field_bigfile
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
from choose_parameters import load_dict
import fitsio

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# save data
def save_pos(pos,type_pos,data_dir,value=None):
    if value is not None:
        data = np.empty(pos.shape[0], dtype=[('Position', ('f8', 3)), ('Value', 'f8')])
        data['Value'] = value
    else:
        data = np.empty(pos.shape[0], dtype=[('Position', ('f8', 3))])
    data['Position'] = pos
    
    # write to a FITS file using fitsio
    fitsio.write(data_dir+"pos_"+type_pos+".fits", data, extname='Data')
    return


def get_positions():
    
    machine = 'alan'
    #machine = 'NERSC'

    #sim_name = "AbacusSummit_hugebase_c000_ph000"
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
    
    # names of the 5 fields tuks
    field_names = ['delta', 'delta_sq', 'nabla_sq', 's_sq']
    factors = {'delta': 1, 'delta_sq': 2, 'nabla_sq': 1, 's_sq': 2}
    
    # load the cosmology
    cosmo = ccl.Cosmology(**cosmo_dict)

    # factor to scale the density as suggested in Modi et al.
    D_z_nbody = ccl.growth_factor(cosmo,1./(1+z_nbody))
    D_z_ic = ccl.growth_factor(cosmo,1./(1+z_nbody))
    D_growth = D_z_nbody/D_z_ic

    # I don't think broadcasting makes sense, might somehow be able to allocate the data better with more ranks?
    '''
    if rank == 0:
        print("loading fields")
        fields = {}
        for i in range(len(field_names)):
            fields[field_names[i]] = load_field_bigfile(field_names[i],dens_dir,R_smooth)
    else:
        fields = None
    fields = comm.bcast(fields,root=0)
    '''
    fields = {}
    for i in range(len(field_names)):
        fields[field_names[i]] = load_field_bigfile(field_names[i],dens_dir,R_smooth)
    
    # create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop over all chunks
    for i_chunk in range(n_chunks):
        #if (rank-1) != i_chunk%size: continue # relevant if broadcasting in root though not really
        if rank != i_chunk%size: continue
        
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
            ic_fns = sorted(glob.glob(user_dict['sim_dir']+"ic_box_L%d_%d*"%(Lbox,user_dict['ppd'])))
            snap_fns = sorted(glob.glob(user_dict['sim_dir']+"snap_box_L%d_%d_%03d*"%(Lbox,user_dict['ppd'],user_dict['ind_snap'])))
            fof_fns = sorted(glob.glob(user_dict['sim_dir']+"fof_snap_box_L%d_%d_%03d*.fits"%(Lbox,user_dict['ppd'],user_dict['ind_snap'])))

            print(ic_fns)
            print(snap_fns)
            print(fof_fns)
            # not gonna work with 1024
            lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,i_chunk,n_chunks)

        save_pos(pos_halo,"halo_%03d"%i_chunk,data_dir)
        del pos_halo

        # get i, j, k for position on the density array
        lagr_ijk = (lagr_pos/(Lbox/N_dim)).astype(int)%N_dim
        del lagr_pos

        # save the particles as they are for the ones field
        save_pos(pos_snap,"ones_snap_%03d"%i_chunk,data_dir)
        for key in fields.keys():
            values = (fields[key]*D_growth**factors[key])[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
            save_pos(pos_snap,key+"_snap_%03d"%i_chunk,data_dir,value=values)
        del pos_snap, lagr_ijk
    print("Saved all particle and halo positions")
    del fields

if __name__ == "__main__":

    t1 = time.time()
    get_positions()
    t2 = time.time(); print("t = ",t2-t1)
    
