import numpy as np
import os
import pyccl as ccl
import glob
import matplotlib.pyplot as plt
import time

from tools.compute_fields import load_fields
from tools.power_spectrum import get_all_cross_ps, get_mesh
from tools.read_abacus import read_abacus
from tools.read_gadget import read_gadget
import fitsio

# save data
def save_pos(pos,weight,type_pos,data_dir):
    data = np.empty(pos.shape[0], dtype=[('Position', ('f8', 3)), ('Weight', 'f8')])
    data['Position'] = pos
    data['Weight'] = weight
    
    # write to a FITS file using fitsio
    fitsio.write(data_dir+"pos_"+type_pos+".fits", data, extname='Data')
    return


def get_templates():
    
    machine = 'alan'
    #machine = 'NERSC'

    #sim_code = 'abacus'
    sim_code = 'gadget'
    
    if sim_code == 'abacus':
        # user choices: abacus
        sim_name = "AbacusSummit_hugebase_c000_ph000"
        interlaced = True
        R_smooth = 2.
        N_dim = 2304 # particle mesh size; usually ppd
        ppd = 2304 # particle per dimension in the sim
        z_nbody = 1. # redshift where we measure power spectrum
        Lbox = 2000. # box size of the simulation [Mpc/h]
        n_chunks = 20

        # cosmological parameters: abacus
        h = 0.6736
        n_s = 0.9649
        Omega_b = 0.02237/h**2
        Omega_c = 0.12/h**2
        sigma8_m = 0.807952

        if machine == 'alan':
            data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/"
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/"

    elif sim_code == 'gadget':
        # user choices: gadget
        sim_name = 'Sim256'
        interlaced = True
        R_smooth = 2.
        ind_snap = 0; z_nbody = 1.
        N_dim = 256 # particle mesh size; usually ppd
        Lbox = 175.#Mpc/h
        ppd = 256
        n_chunks = 1

        # cosmological parameters: gadget
        n_s = 0.96
        Omega_c = 0.655
        Omega_b = 0.045
        h = 0.7
        sigma8_m = 0.8
        
        if machine == 'alan':
            sim_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
        elif machine == 'NERSC':
            sim_dir = "/global/cscratch1/sd/damonge/NbodySims/"+sim_name+"/"
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/"        

    # names of the 5 fields
    field_names = ['ones', 'delta', 'delta_sq', 'nabla_sq', 's_sq']

    # load the cosmology
    cosmo = ccl.Cosmology(n_s=n_s, sigma8=sigma8_m, h=h, Omega_c=Omega_c, Omega_b=Omega_b)

    # factor to scale the density as suggested in Modi et al.
    D_z = ccl.growth_factor(cosmo,1./(1+z_nbody))

    # load the 5 fields
    fields = load_fields(dens_dir,R_smooth,N_dim,Lbox)
    print("Loaded all fields")

    # create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop over all chunks
    N_halo_per_file = np.zeros(n_chunks,dtype=int)
    for i_chunk in range(n_chunks):
        if os.path.exists(data_dir+"pos_s_sq_snap_%03d.fits"%i_chunk):
            print("Data from chunk %d already saved, moving on"%i_chunk)
            continue
        
        if sim_code == 'abacus':
            # load simulation information 
            lagr_pos, pos_snap, halo_table, header = read_abacus(sim_name,z_nbody,i_chunk)
            pos_halo = halo_table['x_L2com']

            # convert [-Lbox/2.,Lbox/2.] to [0,Lbox]
            pos_halo += Lbox/2.
            pos_snap += Lbox/2.
            lagr_pos += Lbox/2.
            
        elif sim_code == 'gadget':
            # find all files
            ic_fns = sorted(glob.glob(sim_dir+"ic_*"))
            snap_fns = sorted(glob.glob(sim_dir+"snap_*"))
            fof_fns = sorted(glob.glob(sim_dir+"fof_*.fits"))
            lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)

        save_pos(pos_halo,np.ones(pos_halo.shape[0]),"halo_%03d"%i_chunk,data_dir)
        N_halo_per_file[i_chunk] = pos_halo.shape[0]
        del pos_halo
            
        # get i, j, k for position on the density array
        lagr_ijk = ((lagr_pos/Lbox)*N_dim).astype(int)%N_dim
        del lagr_pos
        
        for key in fields.keys():
            weights = (fields[key]*D_z)[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
            weights /= np.sum(weights)
            save_pos(pos_snap,weights,key+"_snap_%03d"%i_chunk,data_dir)
        del pos_snap, lagr_ijk

    if np.sum(N_halo_per_file) > 0:
        np.save(data_dir+"N_halo_lengths.npy",N_halo_per_file)
    print("Saved all particle and halo positions")
    del fields

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
    
