import numpy as np
import glob
import time
from memory_profiler import memory_usage

from nbodykit.lab import *
#from nbodykit.utils import ScatterArray # can't broadcast, only parts

from tools.power_spectrum import get_all_cross_ps, get_mesh
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
from choose_parameters import load_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    # redshift choice
    z_nbody = 1.1
    
    #machine = 'alan'
    machine = 'NERSC'

    sim_name = "AbacusSummit_hugebase_c000_ph000"
    #sim_name = 'Sim256'

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
    dk = user_dict['dk']

    field_names = ['ones', 'delta', 'delta_sq', 'nabla_sq', 's_sq']
    
    # get a mesh list for all 5 cases
    mesh_list = []
    for key in field_names:
        pos_snap_fns = sorted(glob.glob(data_dir+"pos_"+key+"_snap_*"))
        # TESTING
        print(pos_snap_fns)
        print(len(pos_snap_fns))
        mesh = get_mesh(pos_snap_fns,N_dim,Lbox,interlaced)
        mesh_list.append(mesh)
    print("Obtained mesh lists for all fields")
    
    # compute all cross power spectra
    ks_all, Pk_all, k_lengths = get_all_cross_ps(mesh_list,dk=dk)
    del mesh_list
    print("Computed cross power spectra of all fields")
    
    # save all power spectra
    np.save(data_dir+"ks_all.npy",ks_all)
    np.save(data_dir+"Pk_all_%d.npy"%(int(R_smooth)),Pk_all)
    np.save(data_dir+"k_lengths.npy",k_lengths)
    print("Saved all templates")

if __name__ == "__main__":

    t1 = time.time()
    main()
    #mem_usage = memory_usage(main(),interval=1., timeout=None)
    #print('Maximum memory usage: %s MB' % np.max(mem_usage))
    t2 = time.time(); print("t = ",t2-t1)
    
