import numpy as np
import numpy.linalg as la
import pyccl as ccl
import os
from numba import jit
import numba as nb 

def load_fields(dens_dir,R_smooth,N_dim,Lbox):
    # load the density field
    density_ic = np.load(dens_dir+"density.npy")
    
    # smooth field
    if os.path.exists(dens_dir+"density_smooth_%d.npy"%(int(R_smooth))):
        density_smooth = np.load(dens_dir+"density_smooth_%d.npy"%(int(R_smooth)))
    else:
        density_smooth = get_smooth_density(density_ic,R=R_smooth,N_dim=N_dim,Lbox=Lbox)
        np.save(dens_dir+"density_smooth_%d.npy"%(int(R_smooth)),density_smooth)

    # release that piece of memory
    del density_ic
    
    # the fields are i = {1,delta,delta^2,nabla^2 delta,s^2} 
    ones = np.ones([N_dim,N_dim,N_dim])
    delta = density_smooth

    if os.path.exists(dens_dir+"delta_sq_%d.npy"%(int(R_smooth))):
        delta_sq = np.load(dens_dir+"delta_sq_%d.npy"%(int(R_smooth)))
    else:
        # compute field
        delta_sq = delta**2
        # subtract mean
        delta_sq -= np.mean(delta_sq)
        np.save(dens_dir+"delta_sq_%d.npy"%(int(R_smooth)),delta_sq)

    fields_missing = []
    if os.path.exists(dens_dir+"nabla_sq_%d.npy"%(int(R_smooth))):
        nabla_sq = np.load(dens_dir+"nabla_sq_%d.npy"%(int(R_smooth)))
    else:
        fields_missing.append('nabla_sq')

    if os.path.exists(dens_dir+"s_sq_%d.npy"%(int(R_smooth))):    
        s_sq = np.load(dens_dir+"s_sq_%d.npy"%(int(R_smooth)))
    else:
        fields_missing.append('s_sq')

    # compute fields
    if len(fields_missing) > 0:
        fields_dict = get_nabla_sq_s_sq(delta, Lbox, N_dim, fields=fields_missing)
        try:
            nabla_sq = fields_dict['nabla_sq']
            nabla_sq -= np.mean(nabla_sq)
            np.save(dens_dir+"nabla_sq_%d.npy"%(int(R_smooth)),nabla_sq)
        except:
            pass
        try:
            s_sq = fields_dict['s_sq']
            s_sq -= np.mean(s_sq)
            np.save(dens_dir+"s_sq_%d.npy"%(int(R_smooth)),s_sq)
        except:
            pass

    # TESTING VERY IMPORTANT ADDED - SIGN
    fields_dict = {'ones': ones, 'delta': delta, 'delta_sq': delta_sq, 'nabla_sq': nabla_sq, 's_sq': s_sq}
    #fields_dict = {'ones': ones, 'delta': delta, 'delta_sq': delta_sq, 'nabla_sq': -nabla_sq, 's_sq': s_sq}
    factors_dict = {'delta': 1., 'delta_sq': 2., 'nabla_sq': 1., 's_sq': 2.}
    return fields_dict, factors_dict

def get_nabla_sq_s_sq(delta,Lbox,N_dim,fields):
    # construct wavenumber array
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    dfour = np.fft.fftn(delta)

    # fields to return
    returned_fields = {}
    if 'nabla_sq' in fields:
        nablasqfour = get_nabla_sq(dfour,karr,N_dim)
        nabla_sq = np.real(np.fft.ifftn(nablasqfour))
        returned_fields['nabla_sq'] = nabla_sq

    if 's_sq' in fields:
        tfour = get_tidal_field(dfour,karr,N_dim)
        tidal_field = np.real(np.fft.ifftn(tfour, axes = (0, 1, 2)))
        s_sq = np.sum(tidal_field**2,axis=(3,4))
        returned_fields['s_sq'] = s_sq

    return returned_fields

def get_nabla_sq(dfour,karr,N_dim):
    # construct ksq
    ksq = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq[:,:,:] = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2

    # compute nabla squared of delta in Fourier space
    nablasqfour = -ksq*dfour
    
    return nablasqfour

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
    
