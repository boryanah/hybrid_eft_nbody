#!/usr/bin/env python3
'''
This is a script for obtaining smooth templates that combine N-body simulations with velocileptors LPT code.

Usage:
------
./obtain_theory.py --z_nbody 0.8
'''

import time
import sys
import os

import asdf
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pyccl as ccl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# home directory
home = os.path.expanduser("~")
sys.path.append(home+'/repos/velocileptors')
from velocileptors.LPT.cleft_fftw import CLEFT
from choose_parameters import load_dict
from tools.read_params import get_dict


# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)

# colour table in HTML hex format
hexcols = np.array(['#44AA99', '#117733', '#999933', '#88CCEE', '#332288', '#BBBBBB', '#4477AA',
           '#CC6677', '#AA4499', '#6699CC', '#AA4466', '#882255', '#661100',
            '#0099BB', '#DDCC77'])
np.random.seed(100)
inds = np.arange(len(hexcols), dtype=int)
np.random.shuffle(inds)
print(inds)
hexcols = hexcols[inds]

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_nbody'] = 0.5#1.1
DEFAULTS['z_ic'] = 99.
DEFAULTS['R_smooth'] = 0.
#DEFAULTS['machine'] = 'NERSC'
DEFAULTS['machine'] = 'alan'

def extrapolate(Pk_tmp, ks, kv, is_pivot, offset=10):
    slope = (np.log10(Pk_tmp[is_pivot+offset])-np.log10(Pk_tmp[is_pivot]))/(np.log10(ks[is_pivot+offset])-np.log10(ks[is_pivot]))
    Pk_0 = 10.**(np.log10(kv[0])*slope + np.log10(Pk_tmp[is_pivot]))
    print(Pk_0,kv[0],slope,ks[is_pivot+offset])
    Pk_frank = np.hstack((Pk_0,Pk_tmp[is_pivot:]))
    kf = np.hstack((0.3*ks[is_pivot],ks[is_pivot:]))
    return Pk_frank, kf

def save_asdf(data_dict, filename, save_dir, header=None):
    # create data tree structure
    data_tree = {}
    data_tree['data'] = data_dict
    if header != None:
        data_tree['header'] = header

    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    output_file.write_to(os.path.join(save_dir,filename))
    output_file.close()


def main(sim_name, z_nbody, z_ic, R_smooth, machine):
    
    # now we need the choose parameters
    if machine == 'alan':
        #data_dir =  home+"/repos/hybrid_eft_nbody/data/%s/z%4.3f/r_smooth_%d/"%(sim_name,z_nbody,int(R_smooth))
        data_dir =  home+"/repos/hybrid_eft_nbody/data/%s/z%4.3f/"%(sim_name,z_nbody)
    else:
        user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)
        data_dir = user_dict['data_dir']

    # indices for the CLASS files
    zs_pk = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 99.0])
    i_pk = np.argmin(np.abs(zs_pk-z_nbody))+1
    i_zel = np.argmin(np.abs(zs_pk-z_ic))+1

    # growth factor
    class_dict = get_dict(sim_name)
    h = np.float(class_dict['h'])
    n_s = np.float(class_dict['n_s'])
    Omega_b = np.float(class_dict['omega_b']/h**2)
    Omega_c = np.float(class_dict['omega_cdm']/h**2)
    sigma8 = np.float(class_dict['sigma8_m'])
    cosmo_dict = {'h': h,
                  'n_s': n_s,
                  'Omega_c': Omega_c,
                  'Omega_b': Omega_b,
                  'sigma8': sigma8}
    cosmo = ccl.Cosmology(**cosmo_dict)
    D_z_nbody = ccl.growth_factor(cosmo, 1./(1+z_nbody))
    D_z_ic = ccl.growth_factor(cosmo, 1./(1+z_ic))
    D_growth = D_z_nbody/D_z_ic

    # name of folder in AbacusSummit/Cosmologies
    index_cosm = int((sim_name.split('c')[-1]).split('_ph')[0])
    name_cosm = "abacus_cosm%03d"%index_cosm
    
    # power spectrum file
    # Lehman's computation
    #pk_fn = home+"/repos/hybrid_eft_nbody/data/%s/z%4.3f/r_smooth_%d/power_nfft2048.csv"%(sim_name,z_nbody,int(R_smooth))
    pk_fn = home+"/repos/AbacusSummit/Cosmologies/"+name_cosm+"/"+name_cosm+".z%d_pk_cb.dat"%i_pk
    pk_zel_fn = home+"/repos/AbacusSummit/Cosmologies/"+name_cosm+"/"+name_cosm+".z%d_pk_cb.dat"%i_zel
                                                                                     
    # templates from nbody
    ks_all = np.load(data_dir+"ks_all.npy")
    Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
    k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
    k_starts = np.zeros(len(k_lengths),dtype=int)
    k_starts[1:] = np.cumsum(k_lengths)[:-1]
    fields = ['1', 'b_1', 'b_2', 'b_{\\nabla^2}', 'b_s']
    
    # scaling initial power spectrum to that redshift:
    z,D,f      = z_nbody, 1., 1.
    klin,plin  = np.loadtxt(pk_fn, unpack=True)
    kzel,pzel  = np.loadtxt(pk_zel_fn, unpack=True)
    # Lehman's computation
    #bs,bs,klin,plin,bs  = np.loadtxt(pk_fn, unpack=True)
    plin      *= D**2
    pzel      *= D_growth**2

    # Initialize the class -- with no wisdom file passed it will
    # experiment to find the fastest FFT algorithm for the system.
    # B.H. modified velocileptors/Utils/loginterp.py in case of exception error
    start = time.time()
    cleft = CLEFT(klin,plin,cutoff=10)
    print("Elapsed time: ",time.time()-start," seconds.")
    # You could save the wisdom file here if you wanted:
    # mome.export_wisdom(wisdom_file_name)

    # The first four are deterministic Lagrangian bias up to third order
    # While alpha and sn are the counterterm and stochastic term (shot noise)
    cleft.make_ptable()
    kv = cleft.pktable[:,0]

    # frankenstein
    k_frank = np.logspace(np.log10(kv[0]),np.log10(kv[-1]),1000)
    
    # parsing the velocileptors spectra
    '''
    r'$(1,b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2, \
    r'$(b_1,b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2, r'$(b_{\nabla^2},b_{\nabla^2})$':cleft.pktable[:,13]*kv**2,
    r'$(b_{\nabla^2},b_s)$':0.5*cleft.pktable[:,13]*kv**2, r'$(b_2,b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2}
    '''
    spectra = {r'$(1,1)$':cleft.pktable[:,1],\
               r'$(1,b_1)$':0.5*cleft.pktable[:,2], r'$(b_1,b_1)$': cleft.pktable[:,3],\
               r'$(1,b_2)$':0.5*cleft.pktable[:,4], r'$(b_1,b_2)$': 0.5*cleft.pktable[:,5],  r'$(b_2,b_2)$': cleft.pktable[:,6],\
               r'$(1,b_s)$':0.5*cleft.pktable[:,7], r'$(b_1,b_s)$': 0.5*cleft.pktable[:,8],  r'$(b_2,b_s)$':0.5*cleft.pktable[:,9],\
               r'$(b_s,b_s)$':cleft.pktable[:,10], r'$(1,b_{\nabla^2})$': np.interp(kv,kzel,pzel*kzel**2), \
               r'$(b_1,b_{\nabla^2})$':np.interp(kv,kzel,pzel*kzel**2), r'$(b_{\nabla^2},b_{\nabla^2})$':np.interp(kv,kzel,pzel*kzel**2),
               r'$(b_{\nabla^2},b_s)$':np.interp(kv,kzel,pzel*kzel**2), r'$(b_2,b_{\nabla^2})$':np.interp(kv,kzel,pzel*kzel**2)}

    # parsing the nbody spectra
    nbody_spectra = {}
    name_dic = {}
    counter = 0
    for i in range(len(fields)):
        for j in range(len(fields)):
            if j < i: continue
            label = r'$('+fields[i]+','+fields[j]+r')$'
            start = k_starts[counter]
            size = k_lengths[counter]

            Pk = Pk_all[start:start+size]
            ks = ks_all[start:start+size]
            Pk = Pk[~np.isnan(ks)]
            ks = ks[~np.isnan(ks)]
            nbody_spectra[label] = Pk
            
            counter += 1

    # dictionary for where to plot each power spectrum
    plot_dic ={r'$(1,1)$':1,
               r'$(1,b_1)$':1, r'$(b_1,b_1)$': 2,\
               r'$(1,b_2)$':4, r'$(b_1,b_2)$': 2,  r'$(b_2,b_2)$': 3,\
               r'$(1,b_s)$':4, r'$(b_1,b_s)$': 5, r'$(b_1,b_{\nabla^2})$': 5,  r'$(b_2,b_s)$': 3, r'$(b_s,b_s)$':6,
               r'$(1,b_{\nabla^2})$':4, r'$(b_2,b_{\nabla^2})$': 3, r'$(b_{\nabla^2},b_s)$': 6, r'$(b_{\nabla^2},b_{\nabla^2})$': 6}

    label_dic={r'$(1,1)$': r'$P_{00}(k)$',
               r'$(1,b_1)$': r'$P_{01}(k)$', r'$(b_1,b_1)$': r'$P_{11}(k)$',\
               r'$(1,b_2)$': r'$P_{02}(k)$', r'$(b_1,b_2)$': r'$P_{12}(k)$',  r'$(b_2,b_2)$': r'$P_{22}(k)$',\
               r'$(1,b_s)$': r'$P_{0s}(k)$', r'$(b_1,b_s)$': r'$P_{1s}(k)$', r'$(b_1,b_{\nabla^2})$': r'$P_{1\nabla}(k)$',  r'$(b_2,b_s)$': r'$P_{2s}(k)$', r'$(b_s,b_s)$': r'$P_{ss}(k)$',
               r'$(1,b_{\nabla^2})$': r'$P_{0\nabla}(k)$', r'$(b_2,b_{\nabla^2})$': r'$P_{2\nabla}(k)$', r'$(b_{\nabla^2},b_s)$': r'$P_{s\nabla}(k)$', r'$(b_{\nabla^2},b_{\nabla^2})$': r'$P_{\nabla\nabla}(k)$'}


    # create frankenstein templates
    spectra_frank_dic = {}
    plt.subplots(2,3,figsize=(18,10))
    for i,key in enumerate(spectra.keys()):
        # pivot scale (i.e. k number where we switch between theory and numerics)
        if key in [r'$(1,b_s)$', r'$(b_1,b_s)$']:
            kpivot = 0.2
        elif key in [r'$(b_2,b_s)$', r'$(b_s,b_s)$']:
            kpivot = 3.e-2
        elif key == r'$(b_{\nabla^2},b_s)$':
            kpivot = 2.e-1
        elif key == r'$(b_2,b_{\nabla^2})$':
            kpivot = 3.e-1
        elif key == r'$(b_1,b_{\nabla^2})$':
            kpivot = 7.e-2
        else:
            kpivot = 9.e-2

        # indices of the pivot
        iv_pivot = np.argmin(np.abs(kv-kpivot))
        is_pivot = np.argmin(np.abs(ks-kpivot))
        if_pivot = np.argmin(np.abs(k_frank-kpivot))
        
        # get the templates and theory for this spectrum
        Pk_tmp = nbody_spectra[key]
        Pk_lpt = spectra[key]

        # LPT defines 1/2 (delta^2-<delta^2>)
        if key == r'$(b_2,b_2)$':
            Pk_tmp /= 4.
        elif 'b_2' in key:
            Pk_tmp /= 2.

        # those are negative so we make them positive in order to show them in logpsace
        if key in [r'$(b_1,b_s)$', r'$(1,b_s)$']:
            Pk_tmp *= -1 
            Pk_lpt *= -1

        # this term is positive if nabla^2 delta = -k^2 delta, but reason we multiply here is that we use k^2 delta instead and k^2 P_zeldovich
        if key == r'$(b_{\nabla^2},b_s)$':
            Pk_tmp *= -1

        # compute the factor
        factor = spectra[key][iv_pivot]/nbody_spectra[key][is_pivot]
        print("factor for %s = "%key,factor)

        # frankensteining
        kf = np.hstack((kv[:iv_pivot], ks[is_pivot:]))
                
        # exterpolate as a power law (done for all nabla^2 terms)
        if key in [r'$(b_{\nabla^2},b_s)$',r'$(b_{\nabla^2},b_{\nabla^2})$',r'$(b_2,b_{\nabla^2})$',r'$(b_1,b_{\nabla^2})$']:
            print("extrapolating")
            if key == r'$(b_{\nabla^2},b_{\nabla^2})$':
                const = np.mean(Pk_tmp[:is_pivot])
                f = interp1d(ks[:is_pivot], np.ones(is_pivot)*const, bounds_error=False, fill_value=const)
                Pk_frank = np.hstack((f(kv[:iv_pivot]), Pk_tmp[is_pivot:]))
            elif key == r'$(b_{\nabla^2},b_s)$':
                Pk_eft = kv**2*(spectra[r'$(1,b_s)$']*0+spectra[r'$(b_1,b_s)$'])
                # og
                #fac = Pk_eft[iv_pivot]/nbody_spectra[key][is_pivot]
                # TESTING
                fac = 0.4
                Pk_frank = np.hstack(((Pk_eft/fac)[:iv_pivot],(nbody_spectra[r'$(b_{\nabla^2},b_s)$'])[is_pivot:]))
            elif key == r'$(b_2,b_{\nabla^2})$':
                Pk_eft = kv**2*(spectra[r'$(1,b_2)$']*0+spectra[r'$(b_1,b_2)$'])
                # TESTING
                fac = 3.7
                Pk_frank = np.hstack(((Pk_eft/fac)[:iv_pivot],(nbody_spectra[key])[is_pivot:]))
            elif key == r'$(b_1,b_{\nabla^2})$':
                Pk_frank = np.hstack(((kv**2*(spectra[r'$(1,b_1)$']+spectra[r'$(b_1,b_1)$']))[:iv_pivot],(nbody_spectra[r'$(b_1,b_{\nabla^2})$'])[is_pivot:]))
            else:
                # currently not used -- extrapolating with a power law
                Pk_frank, kf = extrapolate(Pk_tmp, ks, kv, is_pivot)
        else:
            # og
            #Pk_frank = np.hstack((Pk_lpt[:iv_pivot], Pk_tmp[is_pivot:]))
            # elegant scheme for interpolating between theory and simulations
            w = (1. - np.tanh(100*(kf-kpivot))) * 0.5
            
            f = interp1d(kv, Pk_lpt, bounds_error=False, fill_value=0.)
            Pk_lpt = f(kf)
            f = interp1d(ks, Pk_tmp, bounds_error=False, fill_value=0.)
            Pk_tmp = f(kf)
            Pk_frank = w * Pk_lpt + (1-w) * Pk_tmp
    

        # interpolate with the values that we want
        f = interp1d(kf, Pk_frank, bounds_error=False, fill_value=0.)
        Pk_frank = f(k_frank)
        
        # gaussian filtering to smooth function
        # TESTING
        #Pk_frank[if_pivot-100:if_pivot+100] = gaussian_filter(Pk_frank[if_pivot-100:if_pivot+100], 10.)
        Pk_frank = gaussian_filter(Pk_frank, 10.)

        # getting back to original sign
        # Note: you can also add all of the nabla^2 templates if you wanna have -k^2 delta, but note that in that case b_nabla^2, b_s is positive! 
        if key in [r'$(b_{\nabla^2},b_s)$', r'$(b_1,b_s)$', r'$(1,b_s)$']:
            #print("Multiply by -1 for real run")
            Pk_frank *= -1.
        
        spectra_frank_dic[key] = Pk_frank
        
        print("----------------------------")
        
    # add the wavenumbers to the dictionary
    spectra_frank_dic['ks'] = k_frank
    
    # save as asdf file
    #save_asdf(spectra_frank_dic,"Pk_templates_%d.asdf"%(int(R_smooth)),data_dir)
    print("NOT SAVING ASDF!")
    
    # plot spectra
    for i,key in enumerate(spectra.keys()):

        Pk_frank = spectra_frank_dic[key]
        Pk_tmp = nbody_spectra[key]
        Pk_lpt = spectra[key]
        plot_no = plot_dic[key]
        
        plt.subplot(2,3,plot_no)
        plt.loglog(k_frank, np.abs(Pk_frank), color=hexcols[i], label=label_dic[key])
        plt.loglog(ks, np.abs(Pk_tmp), ls='--', color=hexcols[i])
        if 'nabla' in key:
            pass#plt.loglog(ks, np.abs(Pk_tmp), ls='--', color=hexcols[i])
        else:
            plt.loglog(kv, Pk_lpt, color=hexcols[i], ls=':')#, label=key)
        #plt.loglog(klin, plin, ls='-', color='y')

        plt.legend(ncol=1)

        if plot_no in [1, 4]:
            plt.ylabel(r'$P_{\alpha \beta}(k) \ [({\rm Mpc}/h)^3]$')
        if plot_no in [4, 5, 6]:
            plt.xlabel(r'$k \ [h/{\rm Mpc}]$')

    plt.savefig("figs/templates_"+sim_name.replace('AbacusSummit_base_','')+"_z%4.3f.pdf"%z_nbody)
    plt.close()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_nbody', help='N-body redshift', type=float, default=DEFAULTS['z_nbody'])
    parser.add_argument('--z_ic', help='N-body initial redshift', type=float, default=DEFAULTS['z_ic'])
    parser.add_argument('--R_smooth', help='Smoothing scale', type=float, default=DEFAULTS['R_smooth'])
    parser.add_argument('--machine', help='Machine name', default=DEFAULTS['machine'])
    args = parser.parse_args()
    args = vars(args)
    main(**args)
