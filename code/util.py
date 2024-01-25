import numpy as np
from numba import njit, prange
from scipy.optimize import brentq
from scipy.stats import beta
from numpy.linalg import norm
from tqdm import tqdm


##############
# Uniform upper bounds
##############

def bootstrap_UB(data, nboot = 1000, delta = 0.1, 
                 delta2 = 1):
    n = data.shape[0]
    qboot, _ = compute_UB_qtl(n = n, 
                    truvec = data.mean(axis = 0), 
                    datadist = data, 
                    delta = delta, 
                    delta2 = delta2,
                    reps = nboot)
    return qboot, _

def locsim_bootstrap_UB(data, alphamax, 
                        nboot = 1000, delta = 0.1, gamma = 0.9):
    n = data.shape[0]
    # Obtain the relevant bootstrap quantile
    qboot, qboot_correction = bootstrap_UB(data, nboot, 
        delta = delta, delta2 = (1 - gamma)*delta)
    empvec = data.mean(axis = 0)
    correction_set = empvec - 2*qboot_correction/np.sqrt(n) <= alphamax
    # Now adjust only over the correction set
    data_correction = data[:,correction_set]
    qboot_ls, _ = bootstrap_UB(data_correction,
                    nboot = nboot,
                    delta = gamma*delta)
    qboot_ls = min(qboot, qboot_ls)
    return qboot_ls

##################
# Quantile computations
#####################

# Workhorse function for bootstrap.
#
# Compute the quantiles of KS distance around truvec, 
# by sampling points from datadist with replacement.
# Generally truvec is a empirical estimate based on 
# a holdout set S_H; a surrogate for the true mean
@njit(parallel=True)
def compute_UB_qtl(n, truvec, datadist, delta = 0.1, delta2 = 1, reps = 100):
    Dns = np.zeros(reps)
    Dns2 = np.zeros(reps)
    for k in prange(reps):
        # Sample with replacement from datadist. That is, sample iid
        # from the empirical dist S_D, a surrogate for the true dist
        sampl = subsample(n, datadist)
        empvec = np.sum(sampl, axis = 0) / n
        # Calculate statistic that leads to upper confidence bounds
        Dns[k] = np.sqrt(n) * np.max( (-1) * (empvec - truvec))
        # Calculate statistic that leads to two-sided bounds
        Dns2[k] = np.sqrt(n) * np.max( np.abs(empvec - truvec))
    # The second delta is for use in locally simultaneous inference, 
    # and is always returned.
    res = np.array([np.quantile(Dns, 1 - delta), 
              np.quantile(Dns2, 1 - delta2)
        ])
    return res

@njit(parallel=True)
def compute_UB_qtl_holdout(n, nholdout, datadist, delta = 0.1, delta2 = 1, reps = 100):
    Dns = np.zeros(reps)
    Dns2 = np.zeros(reps)
    ntotal = datadist.shape[0]
    for k in prange(reps):
        # Pull out the holdout set and compute a surrogate truvec.
        idxh = np.array([1] * nholdout + [0] * (ntotal - nholdout)) > 0
        np.random.shuffle(idxh) 
        hdata, datadist_ = datadist[idxh, :], datadist[~idxh, :]
        truvec = np.sum(hdata, axis = 0) / hdata.shape[0]
        # Sample with replacement from datadist. That is, sample iid
        # from the empirical dist S_D, a surrogate for the true dist
        sampl = subsample(n, datadist_)
        empvec = np.sum(sampl, axis = 0) / n
        # Calculate statistic that leads to upper confidence bounds
        Dns[k] = np.sqrt(n) * np.max( (-1) * (empvec - truvec))
        # Calculate statistic that leads to two-sided bounds
        Dns2[k] = np.sqrt(n) * np.max( np.abs(empvec - truvec))
    # The second delta is for use in locally simultaneous inference, 
    # and is always returned.
    res = np.array([np.quantile(Dns, 1 - delta), 
              np.quantile(Dns2, 1 - delta2)
        ])
    return res

# Sample from a distribution
@njit(parallel=True)
def compute_UB_qtl_popul(n, truvec, dist, delta = 0.1, delta2 = 1, reps = 100):
    Dns = np.zeros(reps)
    Dns2 = np.zeros(reps)
    for k in prange(reps):
        # Sample from dist. 
        sampl = dist(n)
        empvec = np.sum(sampl, axis = 0) / n
        # Calculate statistic that leads to upper confidence bounds
        Dns[k] = np.sqrt(n) * np.max( (-1) * (empvec - truvec))
        # Calculate statistic that leads to two-sided bounds
        Dns2[k] = np.sqrt(n) * np.max( np.abs(empvec - truvec))
    # The second delta is for use in locally simultaneous inference, 
    # and is always returned.
    res = np.array([np.quantile(Dns, 1 - delta), 
              np.quantile(Dns2, 1 - delta2)
        ])
    return res




##################
# Main functions
#####################

# Computes many metrics of interest
def compute_UB_metrics_holdout(n, nholdout, datadist, nboot,
                               tradeoff_datadist, tradeoff_type, alphamax,
                               delta = 0.1, reps = 100, binary = False):
    # Initialize metrics
    miscover = {
        'bootstrap_anywhere': np.zeros(reps),
        'bootstrap_tradeoff': np.zeros(reps),
        'bootstrap_tradeoff_alphamax': np.zeros(reps),
        'bootstrap_selected': np.zeros(reps),
        'DKW_anywhere': np.zeros(reps),
        'DKW_tradeoff': np.zeros(reps),
        'DKW_tradeoff_alphamax': np.zeros(reps),
        'DKW_selected': np.zeros(reps),
        'locsim_anywhere': np.zeros(reps),
        'locsim_tradeoff': np.zeros(reps),
        'locsim_tradeoff_alphamax': np.zeros(reps),
        'locsim_selected': np.zeros(reps),
        'pointwise_anywhere': np.zeros(reps),
        'pointwise_tradeoff': np.zeros(reps),
        'pointwise_tradeoff_alphamax': np.zeros(reps),
        'pointwise_selected': np.zeros(reps)
    }
    conservatism = {
        'bootstrap_anywhere': np.zeros(reps),
        'bootstrap_tradeoff': np.zeros(reps),
        'bootstrap_tradeoff_alphamax': np.zeros(reps),
        'bootstrap_selected': np.zeros(reps),
        'DKW_anywhere': np.zeros(reps),
        'DKW_tradeoff': np.zeros(reps),
        'DKW_tradeoff_alphamax': np.zeros(reps),
        'DKW_selected': np.zeros(reps),
        'locsim_anywhere': np.zeros(reps),
        'locsim_tradeoff': np.zeros(reps),
        'locsim_tradeoff_alphamax': np.zeros(reps),
        'locsim_selected': np.zeros(reps),
        'pointwise_anywhere': np.zeros(reps),
        'pointwise_tradeoff': np.zeros(reps),
        'pointwise_tradeoff_alphamax': np.zeros(reps),
        'pointwise_selected': np.zeros(reps)
    }
    ntotal = datadist.shape[0]
    for k in tqdm(range(reps), leave = False):
        # Make the holdout set index
        idxh = np.array([1] * nholdout + [0] * (ntotal - nholdout)) > 0
        np.random.shuffle(idxh)
        # Pull out the holdout set. 
        # Draw n points iid from the dataset that wasn't held out. 
        hdata, datadist_ = datadist[idxh, :], datadist[~idxh, :]
        datadist_ = subsample(n, datadist_)
        # Pull out the traded-off dataset as well
        tdatadist_ = tradeoff_datadist[~idxh, :]
        tdatadist_ = subsample(n, tdatadist_)
        # Compute a surrogate truvec, and their estimates.
        truvec = hdata.mean(axis = 0)
        empvec = datadist_.mean(axis = 0)
        empvec_t = tdatadist_.mean(axis = 0)
        # Now fill out the fields
        algorithms = ['bootstrap', 'DKW', 'locsim', 'pointwise']
        wheres = ['anywhere', 'tradeoff', 'tradeoff_alphamax', 'selected']
        for alg in algorithms:
            if alg == 'bootstrap':
                UBqtl, _ = bootstrap_UB(datadist_, nboot = nboot, delta = delta)
            elif alg == 'DKW':
                UBqtl = np.sqrt(0.5 * np.log(np.e / delta))
            elif alg == 'locsim':
                UBqtl = locsim_bootstrap_UB(datadist_, alphamax = alphamax, 
                    nboot = nboot, delta = delta, gamma = 0.9)
            # Define the upper bound
            if alg == 'pointwise':
                if binary:
                    UBvec = binom_tail_UB(datadist_, delta)
                else:
                    UBvec = WSR_UB(datadist_, delta, maxiters = 100)
            else:
                UBvec = empvec + UBqtl / np.sqrt(n)
            for whr in wheres:
                dname = alg + '_' + whr
                if whr == 'anywhere':
                    msc = any(truvec > UBvec)
                    cons = np.max(UBvec - truvec)
                elif whr == 'tradeoff':
                    istar = tradeoff_two_vecs(empvec, empvec_t, tradeoff_type)
                    msc = truvec[istar] > UBvec[istar]
                    cons = UBvec[istar] - truvec[istar]
                elif whr == 'tradeoff_alphamax':
                    istar = tradeoff_two_vecs(empvec, empvec_t, tradeoff_type, alphamax)
                    msc = truvec[istar] > UBvec[istar]
                    cons = UBvec[istar] - truvec[istar]
                elif whr == 'selected':
                    truselec = truvec[empvec <= alphamax]
                    UBselec = UBvec[empvec <= alphamax]
                    msc = any(truselec > UBselec)
                    cons = np.max(UBselec - truselec)
                miscover[dname][k] = msc
                conservatism[dname][k] = cons
    return miscover, conservatism

# Computes many metrics of interest
def compute_UB_metrics_truvec_notradeoff(n, truvec, dist, nboot, alphamax,
                                        delta = 0.1, reps = 100, binary = False):
    # Initialize metrics
    miscover = {
        'bootstrap_anywhere': np.zeros(reps),
        'DKW_anywhere': np.zeros(reps),
        'locsim_anywhere': np.zeros(reps),
        'pointwise_anywhere': np.zeros(reps),
        'bootstrap_selected': np.zeros(reps),
        'DKW_selected': np.zeros(reps),
        'locsim_selected': np.zeros(reps),
        'pointwise_selected': np.zeros(reps),
    }
    for k in tqdm(range(reps), leave = False):
        if(callable(dist)):
            # If dist is callable, it represents a distribution of losses, which 
            # can be sampled from. 
            datadist_ = dist(n)
            #print("Generated.")
        else:
            # Otherwise, it's an empirical distribution datadist.
            datadist = dist
            datadist_ = subsample(n, datadist)
        # Estimate truvec
        empvec = datadist_.mean(axis = 0)
        # Now fill out the fields
        algorithms = ['bootstrap', 'DKW', 'locsim', 'pointwise']
        wheres = ['anywhere', 'selected']
        for alg in algorithms:
            if alg == 'bootstrap':
                UBqtl, _ = bootstrap_UB(datadist_, nboot = nboot, delta = delta)
            elif alg == 'DKW':
                UBqtl = np.sqrt(0.5 * np.log(np.e / delta))
            elif alg == 'locsim':
                UBqtl = locsim_bootstrap_UB(datadist_, alphamax = alphamax, 
                    nboot = nboot, delta = delta, gamma = 0.9)
            # Define the upper bound
            if alg == 'pointwise':
                if binary:
                    UBvec = binom_tail_UB(datadist_, delta)
                else:
                    UBvec = WSR_UB(datadist_, delta, maxiters = 100)
            else:
                UBvec = empvec + UBqtl / np.sqrt(n)
            for whr in wheres:
                dname = alg + '_' + whr
                if whr == 'anywhere':
                    msc = any(truvec > UBvec)
                elif whr == 'selected':
                    truselec = truvec[empvec <= alphamax]
                    UBselec = UBvec[empvec <= alphamax]
                    msc = any(truselec > UBselec)
                miscover[dname][k] = msc
    return miscover

##########
# Misc
##########

# Draw a bootstrap subsample
@njit
def subsample(m, data):
  nrow = data.shape[0]
  idx = np.random.randint(0, nrow, m)
  subsamp = data[idx, ]
  return subsamp

def tradeoff_two_vecs(vec1, vec2, tradeoff_type, alphamax = 1):
    # Minimize an aggregate loss
    if tradeoff_type == 'equal_weighted_sum':
        objective = 0.5*vec1 + 0.5*vec2
        istar = np.argmin(objective)
    if tradeoff_type == 'elbow':
        pts = np.stack([vec1, vec2], axis = 1)
        objective = dist_pt2line(pts, 
            np.array([1,0]), 
            np.array([0,1]))
        istar = np.argmax(objective)
    # Gets mad if it's greater than alphamax
    if vec1[istar] > alphamax:
        vec1ok = vec1 <= alphamax
        if vec1ok[0] is True:
            # Get the last True
            rev = vec1ok[::-1]
            istar = len(rev) - np.argmax(rev) - 1
        if vec1ok[0] is False:
            # Get the first True
            istar = np.argmax(vec1ok)
    return(istar)

def dist_pt2line(pt, linept1, linept2):
    b = linept2 - linept1
    bhat = b / np.linalg.norm(b)
    bhat = np.expand_dims(bhat, axis = 1)
    bb = np.matmul(bhat, bhat.T)
    p = pt - linept1
    n = p - np.matmul(p, bb)
    return np.linalg.norm(n, axis = 1)

##############
# Pointwise upper bounds
##############

# input: x is an array of losses, evaluated at n datapoints, at p locations
def binom_tail_UB(x, delta):
    n = x.shape[0]
    p = x.shape[1]
    muhat = x.mean(axis = 0)
    return beta.ppf(1 - delta, n * muhat + 1, n * (1 - muhat))
    
def WSR_UB(x, delta, maxiters):
    n = x.shape[0]
    p = x.shape[1]
    denom = np.expand_dims(1 + np.array(range(1,n+1)), 1)
    muhat = (np.cumsum(x, axis = 0) + 0.5) / denom
    sigma2hat = (np.cumsum((x - muhat)**2, axis = 0) + 0.25) / denom
    sigma2hat[1:, :] = sigma2hat[:-1, :] #doesn't match WSR paper
    sigma2hat[0, :] = 0.25  #doesn't match WSR paper
    nu = np.minimum(np.sqrt(2 * np.log( 1 / delta ) / n / sigma2hat), 1)
    def _Kn(R, i): # increasing in R, should cross zero
        return np.max(np.cumsum(np.log(1 - nu[:,i] * (x[:,i] - R)))) + np.log(delta)
    def UB(i):
        _f = lambda R: _Kn(R, i)
        if _f(1) < 0 or np.sign(_f(1e-10)) == np.sign(_f(1-1e-10)):
            return 1
        return brentq(_f, 1e-10, 1-1e-10, maxiter=maxiters)
    res = [UB(i) for i in range(p)]
    return np.array(res)
