from compaso_halo_catalog import CompaSOHaloCatalog
import os

def read_abacus(sim_name,z_nbody,i_chunk):

    # catalog directory
    cat_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name+"/halos/"
    z_dir = os.path.join(cat_dir,"z%.3f"%z_nbody)
    chunk_fn = os.path.join(z_dir,'halo_info','halo_info_%03d.asdf'%i_chunk)
    
    # load halo catalog and 10% particle subsample
    try:
        cat = CompaSOHaloCatalog(chunk_fn, load_subsamples='AB_all', fields=['x_L2com','N'], unpack_bits = True)
    except:
        cat = CompaSOHaloCatalog(chunk_fn, load_subsamples='A_all', fields=['x_L2com','N'], unpack_bits = True)

    # load the pid, position and lagrangian positions
    #pid = cat.subsamples['pid']
    pos_mat = cat.subsamples['pos']
    lagr_pos = cat.subsamples['lagr_pos']
    
    # halo catalog
    halo_table = cat.halos
    #header = cat.header
    N_halos = len(cat.halos)
    print("N_halos = ",N_halos)

    return lagr_pos, pos_mat, halo_table


def read_halo_abacus(sim_name,z_nbody,i_chunk):

    # catalog directory
    cat_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name+"/halos/"
    z_dir = os.path.join(cat_dir,"z%.3f"%z_nbody)
    chunk_fn = os.path.join(z_dir,'halo_info','halo_info_%03d.asdf'%i_chunk)
    
    # load halo catalog
    cat = CompaSOHaloCatalog(chunk_fn, load_subsamples=False, fields=['x_L2com','N'], unpack_bits = False)
    
    # halo catalog
    halo_table = cat.halos
    #header = cat.header
    N_halos = len(cat.halos)
    print("N_halos = ",N_halos)

    return halo_table
