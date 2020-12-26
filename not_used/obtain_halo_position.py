import numpy as np
import os
import pyccl as ccl
import glob
import time
from memory_profiler import memory_usage

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray # can't broadcast, only parts

from tools.compute_fields import load_field_bigfile
from tools.read_abacus import read_halo_abacus
from tools.read_gadget import read_halo_gadget
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
    fitsio.write(data_dir+"pos_"+type_pos+str_mthr+".fits", data, extname='Data')
    return

def main():
    # redshift of snapshots
    zs = [1., 0.7, 0.3, 0.]
    z_nbody = zs[0]

    machine = 'alan'
    #machine = 'NERSC'

    #sim_name = "AbacusSummit_hugebase_c000_ph000"
    #sim_name = "AbacusSummit_hugebase_c000_ph001"
    sim_name = 'Sim256'
    #sim_name = 'Sim1024'
    
    user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
    data_dir = user_dict['data_dir']
    n_chunks = user_dict['n_chunks']
    z_nbody = user_dict['z_nbody']
    Lbox = user_dict['Lbox']
    
    # create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop over all chunks
    for i_chunk in range(n_chunks):
        #if rank != i_chunk%size: continue
        print("saving chunk number %d out of %d chunks"%(i_chunk,n_chunks))
        
        if user_dict['sim_code'] == 'abacus':
            # load simulation information 
            halo_table = read_halo_abacus(sim_name,z_nbody,i_chunk)
            pos_halo = halo_table['x_L2com']
            m_halo = halo_table['N']*user_dict['m_part']

            # convert [-Lbox/2.,Lbox/2.] to [0,Lbox]
            pos_halo += Lbox/2.
            
        elif user_dict['sim_code'] == 'gadget':
            # fof files
            fof_fns = sorted(glob.glob(user_dict['sim_dir']+"fof_snap_box_L%d_%d_%03d*.fits"%(Lbox,user_dict['ppd'],user_dict['ind_snap'])))
            print(fof_fns)
            
            pos_halo, m_halo = read_halo_gadget(fof_fns,i_chunk,n_chunks)


        save_pos(pos_halo,"halo_%03d"%i_chunk,data_dir,mass=m_halo)
        del pos_halo, m_halo

    print("Saved all halo positions")

if __name__ == "__main__":

    t1 = time.time()
    #main()
    main()
    #mem_usage = memory_usage(main,interval=1., timeout=None)
    #print('Memory usage (in chunks of .1 seconds): %s MB' % mem_usage)
    #print('Maximum memory usage: %s MB' % np.max(mem_usage))
    t2 = time.time(); print("t = ",t2-t1)
    
