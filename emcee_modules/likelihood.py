import numpy as np


class PowerData(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, power,cov):
        """
        Constructor of the HSCLikeModule
        """
        self.power = power
        self.cov = cov
        self.icov = np.linalg.inv(cov)

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
        #e.g. load data from files
