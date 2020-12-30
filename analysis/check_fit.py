import os
import sys

import numpy as np
import argparse
import asdf
import yaml
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()
sys.path.append(os.path.expanduser("~/repos/hybrid_eft_nbody/"))
from solve_power import get_P

DEFAULTS = {}
DEFAULTS['path2config'] = '../config/bias_pk.yaml'

def main(path2config):
    
    # Read params from yaml
    config = yaml.load(open(path2config))
    mode = config['modus_operandi']
    power_params = config['power_params']
    template_params = config['template_params']
    
    # load power spectrum data
    power_dir = os.path.expanduser(power_params['power_dir'])
    tmp_dir = os.path.expanduser(template_params['template_dir'])
    z = power_params['z']
    fields = template_params['fields']
    R_smooth = template_params['R_smooth']
    
    # todo: change to sacc
    if mode == 'Pk':
        ks = np.load(os.path.join(power_dir,"ks.npy"))
        Pk_gg = np.load(os.path.join(power_dir,"pk_gg_z%4.3f.npy"%z))
        cov = np.load(os.path.join(power_dir,"cov_pk_gg_z%4.3f.npy"%z))

        # cut in k-space todo: put in function
        k_cut = (ks < power_params['kmax']) & (ks >= power_params['kmin'])
        Pk_gg = Pk_gg[k_cut]
        ks = ks[k_cut]
        cov = cov[:,k_cut]
        cov = cov[k_cut,:]
        Pk_gg_err = np.sqrt(np.diag(cov))
        
        # templates, theory
        Pk_tmps = asdf.open(os.path.join(tmp_dir,'z%.3f'%z,"Pk_templates_%d.asdf"%int(R_smooth)))['data']

        Pk_ij = np.zeros((len(fields),len(fields),len(ks)))
        c = 0
        for i in range(len(fields)):
            for j in range(len(fields)):
                if i > j: continue
                Pk_tmp = Pk_tmps[r'$('+fields[i]+','+fields[j]+r')$']
                Pk_tmp = np.interp(ks,Pk_tmps['ks'],Pk_tmp)

                Pk_ij[i,j,:] = Pk_tmp
                if i != j: Pk_ij[j,i,:] = Pk_tmp
                c += 1

        power_ij = Pk_ij
        power_gg_err = Pk_gg_err
        x = ks
        power_gg = Pk_gg

    elif mode ==  'Cl':
        ells = np.load(os.path.join(power_dir,"ells.npy"))
        Cl_gg = np.load(os.path.join(power_dir,"cl_gg_z%4.3f.npy"%z))
        cov = np.load(os.path.join(power_dir,"cov_cl_gg_z%4.3f.npy"%z))
        Cl_gg_err = np.sqrt(np.diag(cov))

        # templates, theory
        Cl_tmps = asdf.open(os.path.join(tmp_dir,'z%.3f'%z,"Cl_templates_%d.asdf"%int(R_smooth)))['data']
        ells = Cl_tmps['ells']
        Cl_ij = np.zeros((len(fields),len(fields),len(ells)))
        c = 0
        for i in range(len(fields)):
            for j in range(len(fields)):
                if i > j: continue
                Cl_tmp = Cl_tmps[r'$('+fields[i]+','+fields[j]+r')$']
                Cl_tmp = np.interp(ells,Cl_tmps['ells'],Cl_tmp)

                Cl_ij[i,j,:] = Cl_tmp
                if i != j: Cl_ij[j,i,:] = Cl_tmp
                c += 1

        power_ij = Cl_ij
        power_gg_err = Cl_gg_err
        x = ells
        power_gg = Cl_gg

    # best solution from chains
    #F_best = np.array([1., -0.08, -0.34, 0.580, 0.24])
    #F_best = np.array([1., 1., 1., 1., 1.])
    #F_best = np.array([ 1., 1.57785673, -0.85011509, -0.13293844 ,-5.20912406])
    #F_best = np.array([ 1.,   0.99999853,  0.60774282, -0.99997625,  4.9999987])
    #F_best = np.array([ 1.        ,  0.29236532, -0.63308344,  0.62056147,  0.01030821])
    #F_best = np.array([ 1.        , -0.12947702, -0.11833105, -1.14102289, -0.16499079]) # cl
    #F_best = np.array([ 1.        , -0.05398823, -0.82415604,  0.46853151, -0.28520872])
    #F_best = np.array([ 1.        , -0.04971766, -0.91626549,  0.45250239, -0.13577488])
    #F_best = np.array([ 1.        ,  1.88677959, -2.05659463, -1.27924613,  4.91586557])
    F_best =  np.array([ 1.        , -0.04009133, -1.02421888,  0.39142809,  0.45214603])
    
    # obtain the prediction for the gg power spectrum
    power_gg_best, power_gm_best, P_hat = get_P(power_ij, F_best, len(x))

    # plot solution
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    fig.suptitle(r"Bias parameters: %.2f, %.2f, %.2f, %.2f"%(F_best[1],F_best[2],F_best[3],F_best[4]))
    plt.subplots_adjust(left=0.1,right=0.95,top=0.90,bottom=0.15,wspace=0.15)
    plt.subplot(1, 2, 1)
    fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
    
    plt.errorbar(x, power_gg, yerr=power_gg_err, color='black', label='truth', zorder=1)
    plt.plot(x, power_gg_best, color='dodgerblue', label='fit', zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    if mode == 'Pk':
        plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
        plt.ylabel(r"$P(k)$")
    elif mode == 'Cl':
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell$")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, power_gg_best/power_gg-1., color='k')
    plt.axhline(0.)
    plt.ylim([-0.4,0.4])
    plt.xscale('log')
    if mode == 'Pk':
        plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    elif mode == 'Cl':
        plt.xlabel(r"$\ell$")
    plt.savefig("../figs/power_gg_fit.png")
    plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)
