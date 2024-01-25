import numpy as np
from util import compute_UB_qtl_popul
from loss_functions import gaussian_ecdf_dist, generate_equicorr
from scipy.stats import norm
from numba import njit
import argparse

def main(args):
    nsim = args.nsim
    ngrid = args.ngrid
    p = 5
    tgrid = np.linspace(-3, 3, ngrid)
    rhos = np.arange(-0.2, 1, 0.4)
    ns = 3*np.array([10**1, 10**2, 10**3])

    quantiles = []
    for rho in rhos:
        # the true Risk is known in this case
        Risk = norm.cdf(tgrid)
        # Compute KS statistic quantiles for various values of n
        Q = np.zeros(len(ns))
        Sigma = generate_equicorr(p, rho)
        for i, n in enumerate(ns):
            # Specify data generating distribution
            dist = lambda x: gaussian_ecdf_dist(x, p, tgrid, rho, Sigma = Sigma)
            dist = njit(dist)
            Q[i], _ = compute_UB_qtl_popul(n = n, 
                    truvec = Risk, 
                    dist = dist, 
                    delta = 0.1, 
                    reps = nsim)
        quantiles.append(Q)

    kwargs = {}
    for i, rho in enumerate(rhos):
        kwargs['dependence_' + str(np.round(rho, 1))] = quantiles[i]

    np.savez('../results/gaussian_cdf_quantiles.npz', 
            ns = ns, 
            tgrid = tgrid, 
            **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    parser.add_argument("--ngrid", dest = 'ngrid', type = int, help="Number of grid points")
    args = parser.parse_args()
    main(args)


