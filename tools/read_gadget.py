from nbodykit.lab import *
import nbodykit
import glob
import numpy as np
import sys as sys
import matplotlib.pyplot as plt
from astropy.io import fits

def read_grid(prefix,nfiles) :
    f=open(prefix+".0000","rb")
    num_grids,ngrid=np.fromfile(f,dtype=np.int32,count=2)
    f.close()

    print("Will read %d fields"%num_grids+" with %d^3 nodes"%ngrid)
    grid_out=np.zeros([ngrid,ngrid,ngrid,num_grids])
    for ifil in np.arange(nfiles) :
        f=open(prefix+".%04d"%ifil,"rb")
        nug,ng=np.fromfile(f,dtype=np.int32,count=2)
        if (nug!=num_grids) or (ng!=ngrid) :
            print("shit")
            sys.exit(1)
        nx_here=np.fromfile(f,dtype=np.int32,count=1)[0]
        print("File #%d"%(ifil+1)+", %d slices found"%nx_here)
        for ix in np.arange(nx_here) :
            ix_this=np.fromfile(f,dtype=np.int32,count=1)[0]
            grid_out[ix_this,:,:,:]=np.fromfile(f,dtype=np.float32,count=ng*ng*nug).reshape([ng,ng,nug])
        f.close()

    if num_grids==1 :
        grid_out=grid_out[:,:,:,0]

    return grid_out


def get_density(dens_dir,Lbox,ppd,want_show=False):
    prefix_name = "dt_ic_box_L%d_%d"%(int(Lbox),ppd)
    dens_files = glob.glob(dens_dir+prefix_name+"_dens*")
    nfiles_input = len(dens_files)
    dens = read_grid(dens_dir+prefix_name+"_dens", nfiles_input)

    if want_show:
        ng = len(dens)
        i_slice = int(ng/2)

        fig, ax = plt.subplots()
        cax = ax.imshow(dens[i_slice,:,:],interpolation='none',origin='lower')
        cbar = fig.colorbar(cax)
        plt.savefig("../figs/density.png")
        plt.close()

    return dens

def read_halo_gadget(halo_fns,i_chunk,n_chunks,read_header=False):
    # load the halo positions
    hdul = fits.open(halo_fns[i_chunk%len(halo_fns)])
    header = hdul[0].header
    data = hdul[1].data
    hdul.close()

    # select the positions
    X = data['PX_CM']
    Y = data['PY_CM']
    Z = data['PZ_CM']
    m_halo = data['MASS']
    del data
    pos_halo = np.vstack((X,Y,Z)).T

    assert (len(halo_fns) == 1 and len(halo_fns) != n_chunks) or len(halo_fns) == n_chunks,"Something's up with your chunks: %d %d"%(len(halo_fns),n_chunks) 
    
    # check if number of chunks is equal to number of files
    if len(halo_fns) != n_chunks:
        len_chunk = pos_halo.shape[0]//n_chunks
        pos_halo = pos_halo[i_chunk*len_chunk:(i_chunk+1)*len_chunk]
        m_halo = m_halo[i_chunk*len_chunk:(i_chunk+1)*len_chunk]
    
    if read_header:
        return pos_halo, m_halo, header
    
    return pos_halo, m_halo

def read_ic(ic_fns,i_chunk,n_chunks,x_inds=None):
    # load the IC catalog
    cat_ic = nbodykit.io.gadget.Gadget1File(ic_fns[i_chunk%len(ic_fns)])
        
    # load IDs and positions of IC
    id_ic = cat_ic['ID'][:]
    assert np.sum(np.arange(id_ic[0],id_ic[-1]+1)-id_ic) == 0, "the ids of the ICs are not sorted properly"
    del id_ic
    pos_ic = cat_ic['Position'][:]
    del cat_ic

    assert (len(ic_fns) == 1 and len(ic_fns) != n_chunks) or len(ic_fns) == n_chunks,"Something's up with your chunks: %d %d"%(len(ic_fns),n_chunks)
    
    # check if number of chunks is equal to number of files
    if len(ic_fns) != n_chunks:
        if x_inds is None:
            len_chunk = pos_ic.shape[0]//n_chunks
            pos_ic = pos_ic[i_chunk*len_chunk:(i_chunk+1)*len_chunk]
        else:
            pos_ic = pos_ic[x_inds]
    
    return pos_ic

def read_snap(snap_fns,i_chunk,n_chunks,want_chunk=False):
    # load the snapshot catalog
    cat_snap = nbodykit.io.gadget.Gadget1File(snap_fns[i_chunk%len(snap_fns)])
    
    # load IDs and positions of snapshot
    id_snap = cat_snap['ID'][:]
    pos_snap = cat_snap['Position'][:]
    del cat_snap
    
    # sort the ids from 1 to N_part and apply that ordering for the positions
    i_sort_snap = np.argsort(id_snap)
    del id_snap
    pos_snap = pos_snap[i_sort_snap]
    del i_sort_snap

    assert (len(snap_fns) == 1 and len(snap_fns) != n_chunks) or len(snap_fns) == n_chunks,"Something's up with your chunks: %d %d"%(len(snap_fns),n_chunks)
    
    # check if number of chunks is equal to number of files
    if len(snap_fns) != n_chunks:
        len_chunk = pos_snap.shape[0]//n_chunks
        if want_chunk:
            Lbox = 175
            print("WE ARE MAKING STUFF UP FOR TESTING ON SIM256")
            size_chunk = Lbox/n_chunks
            x_sel = (pos_snap[:,0] < i_chunk*size_chunk) & (pos_snap[:,0] < (i_chunk+1)*size_chunk)
            inds_sel = np.zeros(len(x_sel),dtype=int)[x_sel]
            pos_snap = pos_snap[inds_sel]
            return pos_snap, inds_sel
        else:
            pos_snap = pos_snap[i_chunk*len_chunk:(i_chunk+1)*len_chunk]
            
            
    return pos_snap


def read_gadget(ic_fns,snap_fns,halo_fns,i_chunk,n_chunks,want_chunk=False):
    if want_chunk:
        pos_snap, x_inds = read_snap(snap_fns,i_chunk,n_chunks,want_chunk)
        pos_ic = read_ic(ic_fns,i_chunk,n_chunks,x_inds)
    else:
        pos_snap = read_snap(snap_fns,i_chunk,n_chunks,want_chunk)
        pos_ic = read_ic(ic_fns,i_chunk,n_chunks)
    pos_halo, m_halo = read_halo_gadget(halo_fns,i_chunk,n_chunks)
    return pos_ic, pos_snap, pos_halo, m_halo

# OLD
def read_gadget_all(ic_fns,snap_fns,halo_fns,i_chunk):

    hdul = fits.open(halo_fns[i_chunk])
    header = hdul[0].header
    data = hdul[1].data
    X = data['PX_CM']
    Y = data['PY_CM']
    Z = data['PZ_CM']
    pos_halo = np.vstack((X,Y,Z)).T

    # load the IC and some snapshot
    cat_ic = nbodykit.io.gadget.Gadget1File(ic_fns[i_chunk])
    cat_snap = nbodykit.io.gadget.Gadget1File(snap_fns[i_chunk])

    # load IDs and positions of IC
    id_ic = cat_ic['ID'][:]
    pos_ic = cat_ic['Position'][:]

    # load IDs and positions of snapshot
    id_snap = cat_snap['ID'][:]
    pos_snap = cat_snap['Position'][:]

    # find intersection between two
    #intersect, comm1, comm2 = np.intersect1d(id_ic,id_snap,return_indices=True)

    # reorder so that they are matching [1,2,3,...,Npart-1]
    #pos_ic = pos_ic[comm1]
    #pos_snap = pos_snap[comm2]
    i_sort_snap = np.argsort(id_snap)
    pos_snap = pos_snap[i_sort_snap]
    
    return pos_ic, pos_snap, pos_halo

def main():
    machine = 'NERSC'#'alan'

    Lbox = 175.
    ppd = 256
    
    if machine == 'NERSC':
        directory = "/global/cscratch1/sd/damonge/NbodySims/Sim256/"
        data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"
    elif machine == 'alan':
        directory = "/mnt/gosling1/boryanah/small_box_damonge/"
        data_dir = "/mnt/gosling1/boryanah/small_box_damonge/output/"

    # get the coarse density field
    dens = get_density(directory,Lbox,ppd,want_show=True)
    np.save(data_dir+"density.npy",dens)
    
    # find all files
    ic_fns = sorted(glob.glob(directory+"ic_*"))
    snap_fns = sorted(glob.glob(directory+"snap_*"))
    fof_fns = sorted(glob.glob(directory+"fof_*.fits"))
    
    # select snapshot
    i_chunk = 0

    # return position of the particles and halos
    pos_ic, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,i_chunk)
    np.save(data_dir+"pos_ic.npy",pos_ic)
    np.save(data_dir+"pos_snap.npy",pos_snap)
    np.save(data_dir+"pos_halo.npy",pos_halo)
