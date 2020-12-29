import os
import sys

import numpy as np
from astropy.table import Table
import pyccl as ccl
import asdf
import sacc

sys.path.append(os.path.expanduser("~/repos/hybrid_eft_nbody/"))
from project_Cl import project_Cl
#from obtain_theory import save_asdf

class PowerTheory(object):
    """
    Core Module for calculating HSC clustering cls.
    """
    def __init__(self, mode, cosmo, x, z, template_params, cl_params, power_params):
        """
        Constructor of the HSCCoreModule
        """

        self.mode = mode # Cl or Pk mode
        self.cosmo = cosmo # ccl cosmology object
        self.x = x # ks or ells
        self.z = z # redshift of sample
        self.template_params = template_params # Pk_ij parameters
        self.cl_params = cl_params # Cl projection parameters
        self.power_params = power_params # power spectrum parameters
        
    def compute_theory(self, p):
        """
        Compute theoretical prediction for clustering power spectra.
        """

        # Compute Pk = f_i f_j Pk_ij or Cl = f_i f_j Cl_ij
        theory = np.einsum('i,ij...,j', p.flatten(), self.power_ij, p.flatten())
        return theory
        
    def setup(self):
        """
        Sets up the core module.
        Tasks that need to be executed once per run
        """
        
        # rename for ease
        mode = self.mode
        cosmo = self.cosmo
        template_params = self.template_params
        cl_params = self.cl_params
        power_params = self.power_params

        # redshifts at which we have templates
        z_s = np.sort(np.array(template_params['zs']))
        fields = template_params['fields']
        R_smooth = template_params['R_smooth']
        tmp_dir = os.path.expanduser(template_params['template_dir'])

        if mode == 'Pk':
            # load templates at redshift of interest
            Pk_tmps = Table(asdf.open(os.path.join(tmp_dir,"z%4.3f"%self.z,"Pk_templates_%d.asdf"%int(R_smooth)))['data'])

            # obtain the Pk_ij templates
            Pk_ij = np.zeros((len(fields), len(fields), len(self.x)))
            for i in range(len(fields)):
                for j in range(len(fields)):
                    if i > j: continue
                    Pk_tmp = Pk_tmps[r'$('+fields[i]+','+fields[j]+r')$']
                    Pk_tmp = np.interp(self.x, Pk_tmps['ks'], Pk_tmp)

                    Pk_ij[i,j,:] = Pk_tmp
                    if i != j: Pk_ij[j,i,:] = Pk_tmp

            self.power_ij = Pk_ij
            return

        # Redshift distributions todo: read from saccs
        z_eff = power_params['z_eff']
        nz_s = np.exp(-((z_s-z_eff)/0.05)**2/2)

        # bias parameters, todo: perhaps read bias not currently used
        #cl_params['bg']
        a_s = 1./(1+z_s)
        bz_s = 0.95/ccl.growth_factor(cosmo, a_s)

        # galaxy tracer
        tracer = ccl.NumberCountsTracer(cosmo, has_rsd=cl_params['has_rsd'], dndz=(z_s, nz_s), bias=(z_s, bz_s), mag_bias=cl_params['has_magnification'])

        # load all templates
        Pk_tmps_a = {}
        for k in range(len(a_s)):
            Pk_tmps = Table(asdf.open(os.path.join(tmp_dir,"z%4.3f"%z_s[k],"Pk_templates_%d.asdf"%int(R_smooth)))['data'])
            Pk_tmps_a[k] = Pk_tmps
        ks = Pk_tmps['ks']

        # obtain the Cl_ij templates
        Cl_ij = np.zeros((len(fields), len(fields), len(self.x)))
        #data_dic = {} # for saving the templates
        for i in range(len(fields)):
            for j in range(len(fields)):
                if i > j: continue
                Pk_a_s = np.array([Pk_tmps_a[k][r'$('+fields[i]+','+fields[j]+r')$'] for k in range(len(a_s))])
                ells, Cl_tmp = project_Cl(cosmo, tracer, Pk_a_s, ks, a_s)
                Cl_ij[i,j,:] = np.interp(self.x, ells, Cl_tmp)
                if i != j: Cl_ij[j,i,:] = np.interp(self.x, ells, Cl_tmp)
                #data_dic[r'$('+fields[i]+','+fields[j]+r')$'] = Cl_ij[i,j,:]
        #data_dic['ells'] = self.x
        #save_asdf(data_dic,"Cl_templates_%d.asdf"%int(R_smooth),os.path.join(tmp_dir,"z%4.3f"%self.z))
        
        self.power_ij = Cl_ij
        
