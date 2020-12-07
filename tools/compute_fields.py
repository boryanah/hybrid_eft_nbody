import numpy as np
import numpy.linalg as la
import pyccl as ccl
import os
from numba import jit
import numba as nb
import bigfile

from nbodykit.filters import Gaussian 
from nbodykit.lab import *

def filter_txx(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[0]*k[0]/kk

def filter_txy(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[0]*k[1]/kk

def filter_tzx(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[0]*k[2]/kk

def filter_tyz(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[1]*k[0]/kk

def filter_tyy(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[1]*k[1]/kk

def filter_tzz(k, v):
    kk = sum(ki ** 2 for ki in k) 
    kk[kk == 0] = 1
    return v*k[2]*k[2]/kk 

def filter_nabla(k, v):
    kk = sum(ki ** 2 for ki in k) 
    return v*kk

def filter_mnabla(k, v):
    kk = sum(ki ** 2 for ki in k) 
    return -v*kk 


def get_fields_bigfile(dens_dir,R_smooth,N_dim,Lbox):
    # load the density field
    density_ic = BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')

    # get smoothed density in fourier space
    filter_gauss = Gaussian(R_smooth)
    d_smooth_four = density_ic.apply(filter_gauss, mode='complex', kind='wavenumber')
    del density_ic

    # delta smooth and delta squared
    d_smooth = d_smooth_four.paint(mode='real')
    ArrayMesh(d_smooth,BoxSize=Lbox).save(dens_dir+"delta_%d.bigfile"%int(R_smooth), mode='real', dataset='Field')
    ArrayMesh(d_smooth**2,BoxSize=Lbox).save(dens_dir+"delta_sq_%d.bigfile"%int(R_smooth), mode='real', dataset='Field')
    del d_smooth

    # nabla squared
    nabla_sq = (d_smooth_four.apply(filter_nabla, mode='complex', kind='wavenumber')).paint(mode='real')
    nabla_sq -= np.mean(nabla_sq)
    ArrayMesh(nabla_sq,BoxSize=Lbox).save(dens_dir+"nabla_sq_%d.bigfile"%int(R_smooth), mode='real', dataset='Field')
    del nabla_sq

    # minus nabla squared
    '''
    nabla_sq = (d_smooth_four.apply(filter_mnabla, mode='complex', kind='wavenumber')).paint(mode='real')
    nabla_sq -= np.mean(nabla_sq)
    ArrayMesh(nabla_sq,BoxSize=Lbox).save(dens_dir+"mnabla_sq_%d.bigfile"%int(R_smooth), mode='real', dataset='Field')
    del nabla_sq
    '''
    
    # tidal tensor squared
    s_sq = (d_smooth_four.apply(filter_txx, mode='complex', kind='wavenumber')).paint(mode='real')**2+\
           (d_smooth_four.apply(filter_tyy, mode='complex', kind='wavenumber')).paint(mode='real')**2+\
           (d_smooth_four.apply(filter_tzz, mode='complex', kind='wavenumber')).paint(mode='real')**2+\
           2.*(d_smooth_four.apply(filter_txy, mode='complex', kind='wavenumber')).paint(mode='real')**2+\
           2.*(d_smooth_four.apply(filter_tzx, mode='complex', kind='wavenumber')).paint(mode='real')**2+\
           2.*(d_smooth_four.apply(filter_tyz, mode='complex', kind='wavenumber')).paint(mode='real')**2
    del d_smooth_four
    s_sq -= np.mean(s_sq)
    ArrayMesh(s_sq,BoxSize=Lbox).save(dens_dir+"s_sq_%d.bigfile"%int(R_smooth), mode='real', dataset='Field')
    del s_sq

def load_field_bigfile(field_name,dens_dir,R_smooth):
    mesh = BigFileMesh(dens_dir+field_name+"_%d.bigfile"%R_smooth, mode='real', dataset='Field')
    return mesh.paint(mode='real')


def load_field_chunk_bigfile(field_name, dens_dir, R_smooth, i_chunk, n_chunks, Lbox, padding=30.):
    data = bigfile.File(dens_dir+field_name+"_%d.bigfile"%R_smooth)['Field']
    n_gr = int(np.round(data.size**(1./3)))

    grid_size = Lbox/n_gr
    chunk_size = Lbox/n_chunks
    assert grid_size < chunk_size, "The chunk size must be larger than the cell size"

    # starting and finishing index in the grid
    i1, i2 = (np.array([i_chunk*chunk_size-padding,(i_chunk+1)*chunk_size+padding])//grid_size).astype(int)

    # make sure within box
    i1 %= n_gr
    i2 %= n_gr
    # get coordinates in the box of loaded field
    start = (i1*grid_size)%n_gr
    end = ((i2+1)*grid_size)%n_gr
    # convert to indices in bigfile
    i1 *= n_gr**2
    i2 *= n_gr**2
    if i1 > i2:
        data1 = data[i1:]
        data2 = data[:i2]

        n = len(data1)+len(data2)
        field_chunk = np.zeros(n,dtype=np.float32)

        field_chunk[:len(data1)] = data1
        field_chunk[len(data1):] = data2
        del data1, data2
        field_chunk = field_chunk.reshape((i2+1-i1,n_gr,n_gr))
    else:
        field_chunk = data[i1:i2].reshape((i2+1-i1,n_gr,n_gr))
    
    return field_chunk, start, end

##########################################
##########      OLD CODE      ############
##########################################

def get_fourier_smooth(density,R_smooth,N_dim,Lbox):

    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    d_four = np.fft.fftn(density)
    del density
    
    #dksmo = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq[:,:,:] = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2
    d_four[:,:,:] = Wg(ksq,R_smooth)*d_four

    return karr, ksq, d_four

def get_fields(dens_dir,R_smooth,N_dim,Lbox):
    # load the density field
    density_ic = np.load(dens_dir+"density.npy")

    # get smoothed density in fourier space
    karr, ksq, d_smooth_four = get_fourier_smooth(density_ic,R_smooth,N_dim,Lbox)
    del density_ic

    # delta smooth and delta squared
    d_smooth = np.real(np.fft.ifftn(d_smooth_four))    
    np.save(dens_dir+"delta_%d.npy"%(int(R_smooth)),d_smooth)
    np.save(dens_dir+"delta_sq_%d.npy"%(int(R_smooth)),d_smooth**2)
    del d_smooth

    # nabla squared
    nabla_sq = np.real(np.fft.ifftn(ksq*d_smooth_four))
    nabla_sq -= np.mean(nabla_sq)
    np.save(dens_dir+"nabla_sq_%d.npy"%(int(R_smooth)),nabla_sq)
    del nabla_sq

    # tidal tensor squared
    ksq[0,0,0] = 1.
    s_sq = np.real(np.fft.ifftn(karr[:,None,None]*karr[:,None,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[:,None,None]*karr[None,:,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[:,None,None]*karr[None,None,:]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,:,None]*karr[:,None,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,:,None]*karr[None,:,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,:,None]*karr[None,None,:]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,None,:]*karr[:,None,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,None,:]*karr[None,:,None]*d_smooth_four/ksq))**2+\
           np.real(np.fft.ifftn(karr[None,None,:]*karr[None,None,:]*d_smooth_four/ksq))**2
    del d_smooth_four
    s_sq -= np.mean(s_sq)
    np.save(dens_dir+"s_sq_%d.npy"%(int(R_smooth)),s_sq)
    del s_sq

def load_field(field_name,dens_dir,R_smooth):
    if field_name == 'delta':
        return load_delta(dens_dir,R_smooth)
    if field_name == 'delta_sq':
        return load_delta_sq(dens_dir,R_smooth)
    if field_name == 's_sq':
        return load_s_sq(dens_dir,R_smooth)
    if field_name == 'nabla_sq':
        return load_nabla_sq(dens_dir,R_smooth)

def load_s_sq(dens_dir,R_smooth):
    s_sq = np.load(dens_dir+"s_sq_%d.npy"%(int(R_smooth)))
    return s_sq

def load_nabla_sq(dens_dir,R_smooth):
    nabla_sq = np.load(dens_dir+"nabla_sq_%d.npy"%(int(R_smooth)))
    return nabla_sq

def load_delta_sq(dens_dir,R_smooth):
    delta_sq = np.load(dens_dir+"delta_sq_%d.npy"%(int(R_smooth)))
    return delta_sq

def load_delta(dens_dir,R_smooth):
    delta = np.load(dens_dir+"delta_%d.npy"%(int(R_smooth)))
    return delta
    
# This code has been tested before in D. Alonso, B. Hadzhiyska, and M. Strauss (2014/5)
@jit(nopython=True)
def get_tidal_field(dfour,karr,N_dim):
    tfour = np.zeros(shape=(N_dim, N_dim, N_dim, 3, 3),dtype=nb.complex128)
    
    # computing tidal tensor and phi in fourier space
    # and smoothing using the window functions
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                if (a, b, c) == (0, 0, 0):
                    pass
                else:
                    ksq = karr[a]**2 + karr[b]**2 + karr[c]**2
                    # all 9 components
                    tfour[a, b, c, 0, 0] = karr[a]*karr[a]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 1, 1] = karr[b]*karr[b]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 2, 2] = karr[c]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 1, 0] = karr[a]*karr[b]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 0, 1] = tfour[a, b, c, 1, 0]
                    tfour[a, b, c, 2, 0] = karr[a]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 0, 2] = tfour[a, b, c, 2, 0]
                    tfour[a, b, c, 1, 2] = karr[b]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 2, 1] = tfour[a, b, c, 1, 2]

    return tfour

def get_density(pos,weights=None,N_dim=256,Lbox=205.):
    # x, y, and z position
    p_x = pos[:,0]
    p_y = pos[:,1]
    p_z = pos[:,2]

    if weights is None:
        # total number of objects
        N_p = len(p_x)
        # get a 3d histogram with number of objects in each cell
        D, edges = np.histogramdd(np.transpose([p_x,p_y,p_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]])
        # average number of particles per cell
        D_avg = N_p*1./N_dim**3
    else:
        # get a 3d histogram with total mass of objects in each cell
        D, edges = np.histogramdd(np.transpose([p_x,p_y,p_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]],weights=weights)
        # average mass of particles per cell
        D_avg = np.sum(weights)/N_dim**3
    D /= D_avg
    D -= 1.
        
    return D


def Wg(k2, R):
    return np.exp(-k2*R*R/2.)

def get_smooth_density(D,R=4.,N_dim=256,Lbox=205.,return_lambda=False):
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    dfour = np.fft.fftn(D)
    dksmo = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq[:,:,:] = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2
    dksmo[:,:,:] = Wg(ksq,R)*dfour
    if return_lambda:
        lambda1, lambda2, lambda3 = get_eig(dksmo,karr,N_dim)
        drsmo = np.real(np.fft.ifftn(dksmo))
        return drsmo, lambda1, lambda2, lambda3
    drsmo = np.real(np.fft.ifftn(dksmo))
    return drsmo

@jit(nopython=True)
def get_eig(tidt,N_dim):
    evals = np.zeros(shape=(N_dim, N_dim, N_dim, 3))

    for x in range(N_dim):
        for y in range(N_dim):
            for z in range(N_dim):
                # comute and sort evalues in ascending order, for descending add after argsort()[::-1]
                evals[x, y, z, :], evects = la.eig(tidt[x, y, z, :, :])
                idx = evals[x, y, z, :].argsort()
                evals[x, y, z] = evals[x, y, z, idx]
                #evects = evects[:, idx]

    lambda1 = evals[:,:,:,0]
    lambda2 = evals[:,:,:,1]
    lambda3 = evals[:,:,:,2]

    return lambda1, lambda2, lambda3
    
