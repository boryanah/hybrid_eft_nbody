import os

import numpy as np
import sacc

class PowerData(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, mode, power_params):
        """
        Constructor of the HSCLikeModule
        """
        self.mode = mode
        self.power_params = power_params

    def compute_likelihood(self, theory):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        delta = self.power - theory
        lnprob += np.einsum('i,ij,j',delta, self.icov, delta)
        lnprob *= -0.5

        # Return the likelihood
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob

    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        # load all the power spectrum data for Pk or Cl
        power_params = self.power_params
        mode = self.mode
        z = power_params['z']
        power_dir = os.path.expanduser(power_params['power_dir'])
        
        if mode == 'Pk':
            # todo: change to sacc
            ks = np.load(os.path.join(power_dir,"ks.npy"))
            Pk_gg = np.load(os.path.join(power_dir,"pk_gg_z%4.3f.npy"%z))
            cov = np.load(os.path.join(power_dir,"cov_pk_gg_z%4.3f.npy"%z))

            # cut in k-space todo: put in function
            k_cut = (ks < power_params['kmax']) & (ks >= power_params['kmin'])
            Pk_gg = Pk_gg[k_cut]
            ks = ks[k_cut]
            cov = cov[:,k_cut]
            cov = cov[k_cut,:]

            self.power = Pk_gg
            self.x = ks
            
        elif mode == 'Cl':
            # todo: change to sacc
            ells = np.load(os.path.join(power_dir,"ells.npy"))
            Cl_gg = np.load(os.path.join(power_dir,"cl_gg_z%4.3f.npy"%z))
            cov = np.load(os.path.join(power_dir,"cov_cl_gg_z%4.3f.npy"%z))
            
            # cut in k-space todo: put in function tuks
            ell_cut = (ells < power_params['lmax']) & (ells >= power_params['lmin'])
            Cl_gg = Cl_gg[ell_cut]
            ells = ells[ell_cut]
            cov = cov[:,ell_cut]
            cov = cov[ell_cut,:]

            self.power = Cl_gg
            self.x = ells
            
        self.cov = cov
        self.icov = np.linalg.inv(cov)
        self.z = z
