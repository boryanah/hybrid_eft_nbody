import numpy as np

class PowerTheory(object):
    """
    Core Module for calculating HSC clustering cls.
    """
    def __init__(self, Pk_ij, k_lengths, ks_all, param_mapping, def_params):
        """
        Constructor of the HSCCoreModule
        """

        self.Pk_ij = Pk_ij
        self.k_lengths = k_lengths
        self.ks_all = ks_all
        self.param_mapping = param_mapping
        self.def_params = def_params

    def compute_theory(self, p):
        """
        Compute theoretical prediction for clustering power spectra.
        """
        P_theory = np.zeros(len(self.Pk_ij[0,0,:]))
        for i in range(len(p)):
            for j in range(len(p)):
                P_ij = self.Pk_ij[i,j,:]
                f_i = p[i]
                f_j = p[j]
                P_theory += f_i*f_j*P_ij

        return P_theory
        
    def setup(self):
        """
        Sets up the core module.
        Tasks that need to be executed once per run
        """
