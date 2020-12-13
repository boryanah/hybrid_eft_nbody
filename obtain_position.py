import numpy as np
import os
import pyccl as ccl
import glob
import time
from memory_profiler import memory_usage

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray # can't broadcast, only parts

from tools.compute_fields import load_field_bigfile, load_field_chunk_bigfile
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
from choose_parameters import load_dict
import fitsio

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# save data
def save_pos(pos,type_pos,data_dir,value=None,mass=None):
    dtype = [('Position', ('f8', 3))]
    if value is not None:
        dtype.append(('Value', 'f8'))
    if mass is not None:
        dtype.append(('Mass', 'f8'))
    data = np.empty(pos.shape[0], dtype=dtype)
    
    if value is not None:    
        data['Value'] = value
    if mass is not None:
        data['Mass'] = mass
    data['Position'] = pos
    
    # write to a FITS file using fitsio
    fitsio.write(data_dir+"pos_"+type_pos+".fits", data, extname='Data')
    return

def main():
    # redshift choice
    zs = [1., 0.7, 0.3, 0.]
    z_nbody = zs[3]

    z_nbody = 1.1
    
    #machine = 'alan'
    machine = 'NERSC'

    want_chunk = True
    sim_name = "AbacusSummit_hugebase_c000_ph000"
    #sim_name = 'Sim256'
    #sim_name = 'Sim1024'
    
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
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

    if want_chunk:
        fields = {}
        for i in range(len(field_names)):
            fields[field_names[i]], start_pos, end_pos = load_field_chunk_bigfile(field_names[i], dens_dir, R_smooth, rank, n_chunks, Lbox)
        print("loaded chunkwise fields")
    else:
        fields = {}
        for i in range(len(field_names)):
            fields[field_names[i]] = load_field_bigfile(field_names[i], dens_dir, R_smooth)
        print("loaded fields")    
    
    # create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop over all chunks
    for i_chunk in range(n_chunks):
        if rank != i_chunk%size: continue
        print("saving chunk number %d out of %d chunks"%(i_chunk,n_chunks))
        
        if os.path.exists(data_dir+"pos_s_sq_snap_%03d.fits"%i_chunk):
            print("Data from chunk %d already saved, moving on"%i_chunk)
            continue
        
        if user_dict['sim_code'] == 'abacus':
            # load simulation information 
            lagr_pos, pos_snap, halo_table = read_abacus(sim_name,z_nbody,i_chunk)
            pos_halo = halo_table['x_L2com']
            m_halo = halo_table['N']*user_dict['m_part']

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
            
            lagr_pos, pos_snap, pos_halo, m_halo = read_gadget(ic_fns,snap_fns,fof_fns,i_chunk,n_chunks,want_chunk=want_chunk)

        save_pos(pos_halo,"halo_%03d"%i_chunk,data_dir,mass=m_halo)
        del pos_halo, m_halo

        # offset the positions to match the chunk
        if want_chunk:
            if start_pos < end_pos:
                print("normal chunk",i_chunk)
                lagr_pos[:,0] -= start_pos
            else:
                print("subverted chunk",i_chunk)
                choice1 = (start_pos <= lagr_pos[:,0]) & (lagr_pos[:,0] < Lbox)
                choice2 = (end_pos > lagr_pos[:,0]) & (lagr_pos[:,0] >= 0)  
                print("min max mean = ",np.min(lagr_pos[:,0]),np.max(lagr_pos[:,0]),np.mean(lagr_pos[:,0]))
                lagr_pos[choice1,0] -= start_pos
                lagr_pos[choice2,0] += Lbox-start_pos
                print("min max mean = ",np.min(lagr_pos[:,0]),np.max(lagr_pos[:,0]),np.mean(lagr_pos[:,0]))
                
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
    main()
    #mem_usage = memory_usage(main,interval=1., timeout=None)
    #print('Memory usage (in chunks of .1 seconds): %s MB' % mem_usage)
    #print('Maximum memory usage: %s MB' % np.max(mem_usage))
    t2 = time.time(); print("t = ",t2-t1)
    
