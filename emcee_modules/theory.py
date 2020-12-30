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

COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']
BIAS_PARAM_KEYS = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4']

class PowerTheory(object):
    """
    Core Module for calculating HSC clustering cls.
    """
    def __init__(self, mode, x, z, template_params, cl_params, power_params, default_params, param_mapping):
        """
        Constructor of the HSCCoreModule
        """

        self.mode = mode # Cl or Pk mode
        self.x = x # ks or ells
        self.z = z # redshift of sample
        self.template_params = template_params # Pk_ij parameters
        self.cl_params = cl_params # Cl projection parameters
        self.power_params = power_params # power spectrum parameters
        self.default_params = default_params # power spectrum parameters
        self.param_mapping = param_mapping # dictionary with parameter name and index


    def sort_params(self, p):
        
        # separate the cosmology parameters
        if set(COSMO_PARAM_KEYS) == set(self.default_params.keys()):
            # not varying the cosmology
            cosmo = None
        else:
            cosmo_dic = {}
            for key in COSMO_PARAM_KEYS:
                if key in self.param_mapping.keys():
                    cosmo_dic[key] = p[self.param_mapping[key]]
                elif key in self.default_params.keys():
                    cosmo_dic[key] = self.default_params[key]
                else:
                    print("Not all cosmo params are provided"); exit(0)
            cosmo = ccl.Cosmology(**cosmo_dic)

        # separate the bias parameters
        f = np.ones(len(BIAS_PARAM_KEYS))
        for k, key in enumerate(BIAS_PARAM_KEYS):
            if key in self.param_mapping.keys():
                f[k] = p[self.param_mapping[key]]
            else:
                print("Not all bias params are varied"); exit(0)
       
        # TESTING
        #f = p[:]
        return cosmo, f
        
    def compute_theory(self, p):
        """
        Compute theoretical prediction for clustering power spectra.
        """

        # get cosmology and bias parameters
        cosmo, f = self.sort_params(p)
        
        if self.mode == 'Pk':
            # Compute Pk = f_i f_j Pk_ij or Cl = f_i f_j Cl_ij
            theory = np.einsum('i,ij...,j', f.flatten(), self.power_ij, f.flatten())
            return theory
        
        # if dealing with angular power spectra
        if cosmo != None:
            self.init_cl_ij(cosmo) # todo would need compute_cl_ij and use interpolation
        theory = np.einsum('i,ij...,j', f.flatten(), self.Cl_ij, f.flatten())
        return theory

    def init_cl_ij(self, cosmo):

        # galaxy tracer
        tracer = ccl.NumberCountsTracer(cosmo, has_rsd=self.cl_params['has_rsd'], dndz=(self.z_s, self.nz_s), bias=(self.z_s, self.bz_s), mag_bias=self.cl_params['has_magnification'])

        # templates as a function of redshift
        Pk_tmps_a = self.power_ij
        ks = Pk_tmps_a['ks']
        
        # obtain the Cl_ij templates
        Cl_ij = np.zeros((len(self.fields), len(self.fields), len(self.x)))
        for i in range(len(self.fields)):
            for j in range(len(self.fields)):
                if i > j: continue
                Pk_a_s = np.array([Pk_tmps_a[k][r'$('+self.fields[i]+','+self.fields[j]+r')$'] for k in range(len(self.a_s))])
                ells, Cl_tmp = project_Cl(cosmo, tracer, Pk_a_s, ks, self.a_s)
                Cl_ij[i,j,:] = np.interp(self.x, ells, Cl_tmp)
                if i != j: Cl_ij[j,i,:] = np.interp(self.x, ells, Cl_tmp)        
        self.Cl_ij = Cl_ij
        
    def save_cl_ij(self):
        # dictionary for saving the templates
        data_dic = {}
        for i in range(len(self.fields)):
            for j in range(len(self.fields)):
                if i > j: continue
                data_dic[r'$('+self.fields[i]+','+self.fields[j]+r')$'] = self.Cl_ij[i,j,:]
        data_dic['ells'] = self.x
        save_asdf(data_dic,"Cl_templates_%d.asdf"%int(self.R_smooth),os.path.join(self.tmp_dir,"z%4.3f"%self.z))


        
    def load_cl_ij(self):
        """
        Compute theoretical prediction for the angular power spectra templates
        """
        # if not varying the cosmology
        Cl_data = asdf.open(os.path.join(self.tmp_dir,"z%4.3f"%self.z,"Cl_templates_%d.asdf"%int(self.R_smooth)))['data']
        Cl_ij = np.zeros((len(self.fields), len(self.fields), len(self.x)))
        for i in range(len(self.fields)):
            for j in range(len(self.fields)):
                if i > j: continue
                Cl_ij[i,j,:] = np.array([Cl_data[r'$('+self.fields[i]+','+self.fields[j]+r')$']])
                if i != j: Cl_ij[j,i,:] = Cl_ij[i,j,:]
        return Cl_ij
        
    
    def setup(self):
        """
        Sets up the core module.
        Tasks that need to be executed once per run
        """

        # redshifts at which we have templates
        z_s = np.array(self.template_params['zs'])
        bz_s = np.array(self.template_params['bs'])
        i_sort = np.argsort(z_s)
        self.z_s = z_s[i_sort]
        self.bz_s = bz_s[i_sort]
        self.a_s = 1./(1+self.z_s)
        self.fields = self.template_params['fields']
        self.R_smooth = self.template_params['R_smooth']
        self.tmp_dir = os.path.expanduser(self.template_params['template_dir'])

        if self.mode == 'Pk':
            # load templates at redshift of interest
            Pk_tmps = Table(asdf.open(os.path.join(self.tmp_dir,"z%4.3f"%self.z,"Pk_templates_%d.asdf"%int(self.R_smooth)))['data'])

            # obtain the Pk_ij templates
            Pk_ij = np.zeros((len(self.fields), len(self.fields), len(self.x)))
            for i in range(len(self.fields)):
                for j in range(len(self.fields)):
                    if i > j: continue
                    Pk_tmp = Pk_tmps[r'$('+self.fields[i]+','+self.fields[j]+r')$']
                    Pk_tmp = np.interp(self.x, Pk_tmps['ks'], Pk_tmp)

                    Pk_ij[i,j,:] = Pk_tmp
                    if i != j: Pk_ij[j,i,:] = Pk_tmp

            self.power_ij = Pk_ij
            return

        # Redshift distributions todo: read from saccs
        z_eff = self.power_params['z_eff']
        self.nz_s = np.exp(-((self.z_s-z_eff)/0.05)**2/2)
        
        # load all templates
        Pk_tmps_a = {}
        for k in range(len(self.a_s)):
            Pk_tmps = Table(asdf.open(os.path.join(self.tmp_dir,"z%4.3f"%self.z_s[k],"Pk_templates_%d.asdf"%int(self.R_smooth)))['data'])
            Pk_tmps_a[k] = Pk_tmps
        Pk_tmps_a['ks'] = Pk_tmps['ks']
        self.power_ij = Pk_tmps_a        
