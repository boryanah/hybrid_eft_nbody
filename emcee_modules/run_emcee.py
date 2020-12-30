#! /usr/bin/env python

import os
import time
import sys

import numpy as np
import argparse
import emcee
import yaml
import pyccl as ccl
#from InitializeFromChain import InitializeFromChain

from likelihood import PowerData
from theory import PowerTheory

DEFAULTS = {}
DEFAULTS['path2config'] = os.path.expanduser("~/repos/hybrid_eft_nbody/config/bias_pk.yaml")
COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']

class SampleFileUtil(object):
    """
    Util for handling sample files.
    Copied from Andrina's code.

    :param filePrefix: the prefix to use
    :param reuseBurnin: True if the burn in data from a previous run should be used
    """

    def __init__(self, filePrefix, carry_on=False):
        self.filePrefix = filePrefix
        if carry_on:
            mode = 'a'
        else:
            mode = 'w'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        """
        Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"

class DumPool(object):
    def __init__(self):
        pass

    def is_master(self):
        return True

    def close(self):
        pass

def kmax2lmax(kmax, zeff, cosmo=None):
    """
    Determine lmax corresponding to given kmax at an effective redshift zeff according to
    kmax = (lmax + 1/2)/chi(zeff)
    :param kmax: maximal wavevector in h Mpc^-1
    :param zeff: effective redshift of sample
    :return lmax: maximal angular multipole corresponding to kmax
    """

    if cosmo is None:
        print('CCL cosmology object not supplied. Initializing with Planck 2018 cosmological parameters.')
        cosmo = ccl.Cosmology(n_s=0.9649, A_s=2.1e-9, h=0.6736, Omega_c=0.264, Omega_b=0.0493)

    # Comoving angular diameter distance in Mpc/h
    chi_A = ccl.comoving_angular_distance(cosmo, 1./(1.+zeff))*cosmo['h']
    lmax = kmax*chi_A - 1./2.

    return lmax
    
def time_lnprob(params, Data, Theory):
    print('   ==========================================')
    print("   | Calculating likelihood evaluation time |")
    print('   ==========================================')
    timing = np.zeros(10)
    for i in range(10):
        print('Test ',i,' of 9')
        start = time.time()
        if i<5:
            lnprob(params[:, 0]+i*0.1*params[:, 3], params, Data, Theory)
        else:
            lnprob(params[:, 0]-(i-4)*0.1*params[:, 3], params, Data, Theory)
        finish = time.time()
        timing[i] = finish-start
                
    mean = np.mean(timing)
    print('============================================================================')
    print('mean computation time: ', mean)
    stdev = np.std(timing)
    print('standard deviation : ', stdev)
    print('============================================================================')
    return

def inrange(p, params):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p, params, Data, Theory):
    if inrange(p, params):
        if True:#try:
            theory = Theory.compute_theory(p)
            lnP = Data.compute_likelihood(theory)
        if False:#except: TESTING
            lnP = -np.inf
    else:
        lnP = -np.inf
    return lnP


def main(path2config,time_likelihood):

    # Read params from yaml
    config = yaml.load(open(path2config))
    mode = config['modus_operandi']
    default_params = config['default_params']
    power_params = config['power_params']
    template_params = config['template_params']
    cl_params = config['cl_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        param_mapping[key] = fit_params[key][0]
        params[fit_params[key][0], :] = fit_params[key][1:]
    
    # Cosmology
    if set(COSMO_PARAM_KEYS) == set(default_params.keys()):
        print("We are NOT varying the cosmology")
        #cosmo_dict = {}
        #for key in COSMO_PARAM_KEYS:
        #cosmo_dict[key] = default_params[key]
        cosmo = ccl.Cosmology(**default_params)
    else:
        print("We ARE varying the cosmology")
        cosmo = None
    
    if power_params['lmax'] == 'kmax':
        lmax = kmax2lmax(power_params['kmax'], power_params['z'], cosmo)
        power_params['lmax'] = lmax
    if power_params['lmin'] == 'kmax':
        lmin = 0.
        power_params['lmin'] = lmin

    # read data parameters
    Data = PowerData(mode, power_params)
    Data.setup()
        
    # read theory parameters
    Theory = PowerTheory(mode, Data.x, Data.z, template_params, cl_params, power_params, default_params, param_mapping)
    Theory.setup()

    # initialize the Cl templates
    if mode == 'Cl' and cosmo != None:
        Theory.init_cl_ij(cosmo)

    # Make path to output
    if not os.path.isdir(os.path.expanduser(ch_config_params['path2output'])):
        try:
            os.makedirs(os.path.expanduser(ch_config_params['path2output']))
        except:
            pass

    # MPI option
    if ch_config_params['use_mpi']:
        from schwimmbad import MPIPool
        pool = MPIPool()
        print("Using MPI")
        pool_use = pool
    else:
        pool = DumPool()
        print("Not using MPI")
        pool_use = None
        
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    # just time the likelihood calculation
    if time_likelihood:
        time_lnprob(params, Data, Theory)
        return
    
    # emcee parameters
    nwalkers = nparams * ch_config_params['walkersRatio']
    nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(ch_config_params['path2output']),
                                ch_config_params['chainsPrefix'])

    # fix initial conditions
    found_file = os.path.isfile(prefix_chain+'.txt')
    if (not found_file) or (not ch_config_params['rerun']):
        p_initial = params[:, 0] + np.random.normal(size=(nwalkers, nparams)) * params[:, 3][None, :]
        nsteps_use = nsteps
    else:
        print("Restarting from a previous run")
        old_chain = np.loadtxt(prefix_chain+'.txt')
        p_initial = old_chain[-nwalkers:,:]
        nsteps_use = max(nsteps-len(old_chain) // nwalkers, 0)

    # initializing sampler
    chain_file = SampleFileUtil(prefix_chain, carry_on=ch_config_params['rerun'])
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, args=(params,Data,Theory), pool=pool_use)
    start = time.time()
    print("Running %d samples" % nsteps_use)

    # record every iteration
    counter = 1
    for pos, prob, _ in sampler.sample(p_initial, iterations=nsteps_use):
        if pool.is_master():
            print('Iteration done. Persisting.')
            chain_file.persistSamplingValues(pos, prob)

            if counter % 10:
                print(f"Finished sample {counter}")
        counter += 1

    pool.close()
    end = time.time()
    print("Took ",(end - start)," seconds")


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    parser.add_argument('--time_likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')
    
    args = vars(parser.parse_args())    
    main(**args)
