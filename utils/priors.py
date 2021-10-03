from __future__ import division
import numpy as np
import torch


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out


class laplace_prior(object):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b

    def loglike(self, x, do_sum=True):
        if do_sum:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b).sum()
        else:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b)


class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term)



