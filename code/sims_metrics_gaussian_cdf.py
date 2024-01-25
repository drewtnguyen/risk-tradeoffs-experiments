# Compute coverage and conservatism for tumors
import numpy as np
import argparse
from util import compute_UB_metrics_truvec_notradeoff
from loss_functions import gaussian_ecdf_dist, generate_equicorr
from scipy.stats import norm
from numba import njit


def main(args):
    nsim = args.nsim
    ngrid = args.ngrid
    nboot = args.nboot

    # Set up parameters
    p = 5
    tgrid = np.linspace(-3, 3, ngrid)
    ns = 3*np.array([10**1, 10**2, 10**3])
    nreps = nsim
    rhos = [-0.2, 0.2, 0.6]

    miscovers_by_rho = {
        '-0.2': [], 
        '0.2': [], 
        '0.6': []
    }

    types = [
            'bootstrap_anywhere',
            'DKW_anywhere',
            'locsim_anywhere',
            'pointwise_anywhere',
            'bootstrap_selected',
            'DKW_selected',
            'locsim_selected',
            'pointwise_selected'
    ]

    total_metrics = len(rhos) * len(ns)
    metrics_done = 0
    for rho in rhos:
        Sigma = generate_equicorr(p, rho)
        for i, n in enumerate(ns):
            dist = lambda x: gaussian_ecdf_dist(x, p, tgrid, rho, Sigma = Sigma)
            dist = njit(dist)
            truvec = norm.cdf(tgrid)
            print("Computing metrics for settings " + str(metrics_done) + "/" + str(total_metrics) + ", n = " + str(n))
            metrics_done += 1
            miscover = compute_UB_metrics_truvec_notradeoff(
                n = n, truvec = truvec, dist = dist,
                nboot = nboot,
                alphamax = 0.2,
                delta = 0.1, 
                reps = nreps
                )
            miscovers_by_rho[str(rho)].append(miscover)


    mkwargs = {}

    for rho in rhos:
        for i, n in enumerate(ns):
            for t in types:
                mkwargs['dependence_' + str(rho) + '_' + t + '_' + str(n)] = miscovers_by_rho[str(rho)][i][t]

    mkwargs['ns'] = ns

    np.savez('../results/gaussian_cdf_miscover.npz', **mkwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    parser.add_argument("--ngrid", dest = 'ngrid', type = int, help="Number of grid points")
    parser.add_argument("--nboot", dest = 'nboot', type = int, help="Number of bootstrap samples")
    args = parser.parse_args()
    main(args)

