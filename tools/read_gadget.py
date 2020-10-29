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


def get_density(directory,want_show=False):
    nfiles_input = 1
    prefix_name = "dt_ic_box_L175_256"
    dens = read_grid(directory+prefix_name+"_dens", nfiles_input)

    if want_show:
        ng = len(dens)
        i_slice = int(ng/2)

        fig, ax = plt.subplots()
        cax = ax.imshow(dens[i_slice,:,:],interpolation='none',origin='lower')
        cbar = fig.colorbar(cax)
        plt.savefig("figs/density.png")
        plt.close()

    return dens

def read_gadget(ic_fns,snap_fns,halo_fns,ind_snap):

    hdul = fits.open(halo_fns[ind_snap])
    header = hdul[0].header
    data = hdul[1].data
    X = data['PX_CM']
    Y = data['PY_CM']
    Z = data['PZ_CM']
    pos_halo = np.hstack((X,Y,Z)).T

    
    # load the IC and some snapshot
    cat_ic = nbodykit.io.gadget.Gadget1File(ic_fns[0])
    cat_snap = nbodykit.io.gadget.Gadget1File(snap_fns[ind_snap])

    # load IDs and positions of IC
    id_ic = cat_ic['ID'][:]
    pos_ic = cat_ic['Position'][:]

    # load IDs and positions of snapshot
    id_snap = cat_snap['ID'][:]
    pos_snap = cat_snap['Position'][:]

    # find intersection between two
    intersect, comm1, comm2 = np.intersect1d(id_ic,id_snap,return_indices=True)

    # reorder so that they are matching [1,2,3,...,Npart-1]
    pos_ic = pos_ic[comm1]
    pos_snap = pos_snap[comm2]
    
    # TODO: it is more efficient to just order both in ascending order (in fact IC always ascending)
    
    return pos_ic, pos_snap, pos_halo

def main():
    # directory of simulation
    directory = "/global/cscratch1/sd/damonge/NbodySims/Sim256/"

    # get the coarse density field
    dens = get_density(directory,want_show=True)
    np.save("data/density.npy",dens)
    
    
    # find all files
    ic_fns = sorted(glob.glob(directory+"ic_*"))
    snap_fns = sorted(glob.glob(directory+"snap_*"))
    fof_fns = sorted(glob.glob(directory+"fof_*.fits"))
    
    # select snapshot
    ind_snap = 0

    # return position of the particles and halos
    pos_ic, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)


