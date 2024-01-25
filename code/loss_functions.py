import numpy as np
from njit_multinorm import multivariate_normal
from numba import njit

###################
# Gaussian ECDF "loss"
#####################
@njit
def generate_equicorr(n, rho):
    # Create an nxn matrix filled with the value 'rho'
    cov_matrix = np.full((n, n), rho)
    # Set the diagonal elements to 1
    np.fill_diagonal(cov_matrix, 1.0)
    return cov_matrix

@njit
def compute_ecdf(t, X):
    t = t[:, np.newaxis, np.newaxis]
    indic = X <= t
    res = indic.sum(axis = 2) / indic.shape[2]
    res = res.T
    return res

@njit
def gaussian_ecdf_dist(n, p, tgrid, rho, Sigma = None):
    if Sigma is None:
        Sigma = generate_equicorr(p, rho)
    X = multivariate_normal(mean = np.zeros(p), cov = Sigma, size = n)
    return compute_ecdf(tgrid, X)

###########################
# Binary classification losses
###########################

def type_I_dist(n, tgrid):
    pass 

def type_II_dist(n, tgrid):
    pass 



###################
# Multi-label classification losses
#####################


def FNP(t, sgmd, labels):
    t = t[:, np.newaxis, np.newaxis]
    preds = sgmd > t
    notpreds = 1 - preds
    res = (notpreds * labels).sum(axis=2)/labels.sum(axis=1)
    res = res.T
    return(res)

def FPP(t, sgmd, labels):
    t = t[:, np.newaxis, np.newaxis]
    preds = sgmd > t
    notlabels = 1 - labels
    res = (preds * notlabels).sum(axis=2)/notlabels.sum(axis=1)
    res = res.T
    return(res)

def FDP(t, sgmd, labels):
    t = t[:, np.newaxis, np.newaxis]
    preds = sgmd > t
    notlabels = 1 - labels
    num_disc = preds.sum(axis=2)
    # replace zero denominators with 1
    num_disc[num_disc == 0] = 1
    res = (preds * notlabels).sum(axis=2)/num_disc
    res = res.T
    return(res)

def SetSize(t, sgmd, labels):
    t = t[:, np.newaxis, np.newaxis]
    preds = sgmd > t
    res = preds.mean(axis=2)
    res = res.T
    return(res)

# Monotonization of a matrix of loss values on a gridded t
def monotonize_incr(lossmat):
    if lossmat.ndim == 1:
        lossmat = lossmat[np.newaxis, :]
    res = np.maximum.accumulate(lossmat, axis = 1)
    res = np.squeeze(res)
    return(res)

def monotonize_decr(lossmat):
    if lossmat.ndim == 1:
        lossmat = lossmat[np.newaxis, :]
    res = np.flip(lossmat, axis = 1)
    res = np.maximum.accumulate(res, axis = 1)
    res = np.flip(res, axis = 1)
    res = np.squeeze(res)
    return(res)




