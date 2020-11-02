from compaso_halo_catalog import CompaSOHaloCatalog

def read_abacus(sim_name,z_nbody,i_chunk):

    # catalog directory
    cat_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name+"/halos/"
    catdir = os.path.join(cat_dir,"z%.3f"%z_nbody)
    chunk_fn = os.path.join(catdir,'halo_info','halo_info_%3d.asdf'%i_chunk)
    
    # load halo catalog and 10% particle subsample
    cat = CompaSOHaloCatalog(chunk_fn, load_subsamples='AB_all', fields=['x_L2com'], unpack_bits = True)

    # halo catalog
    halo_table = cat.halos
    header = cat.header
    N_halos = len(cat.halos)
    print("N_halos = ",N_halos)

    # load the pid, position and lagrangian positions
    #pid = cat.subsamples['pid']
    pos_mat = cat.subsamples['pos']
    lagr_pos = cat.subsamples['lagr_pos']

    return lagr_pos, pos_mat, halo_table, header

