#! /usr/bin/env python

import time
import sys

import numpy as np
import os
import asdf
import argparse
import sacc
import pyccl as ccl
import emcee
import yaml
#from InitializeFromChain import InitializeFromChain

from likelihood import PowerData
from theory import PowerTheory

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





def inrange(p):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p):
    if inrange(p):
        try:
            theory = Theory.compute_theory(p)
            lnP = Data.compute_likelihood(theory)
        except:
            lnP = -np.inf
    else:
        lnP = -np.inf
    return lnP


def main(path2config,time_likelihood):

    # Read params from yaml
    config = yaml.load(open(path2config))

    # Make path to output
    ch_config_params = config['ch_config_params']
    if not os.path.isdir(ch_config_params['path2output']):
        try:
            os.makedirs(ch_config_params['path2output'])
        except:
            pass

    # power spectrum directory
    power_dir = config['power_dir'][0]
    power_params = config['power_params']
    zs = power_params['zs']
    # todo: generalize
    z = zs[0]
    z_str = "z%.3f"%z
    power_dir = os.path.join(power_dir,z_str)
    
    ks = np.load(os.path.join(power_dir,"ks.npy"))
    Pk_hh = np.load(os.path.join(power_dir,"Pk_hh.npy"))
    #cov = np.load(os.path.join(power_dir,"covmat_hh.npy"))
    Lbox = config['default_params']['Lbox'] # TESTING CHANGE TODO TUKS
    dk = ks[1]-ks[0]
    k_cut = (ks < power_params['kmax']) & (ks >= power_params['kmin'])
    Pk_hh = Pk_hh[k_cut]
    ks = ks[k_cut]
    # todo: change
    N_modes = ks**2*dk*Lbox**3/(2.*np.pi**2)
    Pk_hh_err = Pk_hh*np.sqrt(2./N_modes)
    Pk_hh_err[0] = 1.e-6    
    cov = np.diag(Pk_hh_err**2)

    # initialize data object
    global Data
    Data = PowerData(Pk_hh, cov)
    Data.setup()
    
    # parameters to fit
    fit_params = config['fit_params']
    nparams = len(fit_params.keys())
    param_mapping = {}
    global params
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        param_mapping[key] = fit_params[key][0]
        params[fit_params[key][0], :] = fit_params[key][1:]

    # templates
    template_params = config['template_params']
    R_smooth = template_params['R_smooth']
    tmp_dir = config['template_dir'][0]
    tmp_dir = os.path.join(tmp_dir,'z%.3f'%z)
    Pk_all = np.load(os.path.join(tmp_dir,"Pk_all_%d.npy"%(int(R_smooth))))
    ks_all = np.load(os.path.join(tmp_dir,"ks_all.npy"))
    Pk_tmps = asdf.open(os.path.join(tmp_dir,"Pk_templates_%d.asdf"%int(R_smooth)))['data']
    k_lengths = np.load(os.path.join(tmp_dir,"k_lengths.npy")).astype(int)
    Pk_all = Pk_all.reshape(int(len(ks_all)/k_lengths[0]),k_lengths[0])
    Pk_all = Pk_all[:,k_cut]
    k_length = Pk_all.shape[1]
    fields_tmp = ['1', 'b_1', 'b_2', 'b_{\\nabla^2}', 'b_s']
    Pk_ij = np.zeros((nparams,nparams,k_length))
    c = 0
    for i in range(nparams):
        for j in range(nparams):
            if i > j: continue
            # TESTING
            Pk_tmp = Pk_tmps[r'$('+fields_tmp[i]+','+fields_tmp[j]+r')$']
            Pk_tmp = np.interp(ks,Pk_tmps['ks'],Pk_tmp)
            # original
            #Pk_tmp = Pk_all[c]
            
            Pk_ij[i,j,:] = Pk_tmp
            if i != j: Pk_ij[j,i,:] = Pk_tmp
            c += 1

    # initialize theory
    global Theory
    Theory = PowerTheory(Pk_ij, k_lengths, ks_all, param_mapping, config['default_params'])
    Theory.setup()

    # bias parameters, not currently used
    #config['default_params'].update(cl_params['bg'])


    if time_likelihood:
        print('   ==========================================')
        print("   | Calculating likelihood evaluation time |")
        print('   ==========================================')
        timing = np.zeros(10)
        for i in range(10):
            print('Test ',i,' of 9')
            start = time.time()
            if i<5:
                lnprob(params[:, 0]+i*0.1*params[:, 3])
            else:
                lnprob(params[:, 0]-(i-4)*0.1*params[:, 3])
            finish = time.time()
            timing[i] = finish-start
                
        mean = np.mean(timing)
        print('============================================================================')
        print('mean computation time: ', mean)
        stdev = np.std(timing)
        print('standard deviation : ', stdev)
        print('============================================================================')
        return
                    
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


    # emcee parameters
    nwalkers = nparams * ch_config_params['walkersRatio']
    nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']

    # where to record
    prefix_chain = os.path.join(ch_config_params['path2output'],
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
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, pool=pool_use)
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', required=True)
    parser.add_argument('--time-likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')
    
    args = vars(parser.parse_args())    
    main(**args)
