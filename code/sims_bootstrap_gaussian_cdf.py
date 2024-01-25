import numpy as np
from util import compute_UB_qtl_popul, bootstrap_UB
from loss_functions import generate_equicorr, gaussian_ecdf_dist
import argparse


def main(args):
    nsim = args.nsim
    ngrid = args.ngrid
    nboot = args.nboot

    # At n = 20000 data, nboot = 1000 about 30 seconds per sim. Doable, but takes forever.

    p = 5
    tgrid = np.linspace(-3, 3, ngrid)
    rhos = np.arange(-0.2, 1, 0.4)
    ns = 3*np.array([10**1, 10**2, 10**3])

    # Save the quantiles of the quantiles
    rhos = [-0.2, 0.2, 0.6]
    quantiles10 = {
        '-0.2': None, 
        '0.2': None, 
        '0.6': None
    }
    quantiles50 = {
        '-0.2': None, 
        '0.2': None, 
        '0.6': None
    }
    quantiles90 = {
        '-0.2': None, 
        '0.2': None, 
        '0.6': None
    }
    # Each simulation run, draw n points iid from the population
    # and compute a new bootstrap quantile. 
    for rho in rhos:
        Q = np.zeros([len(ns), nsim])
        Sigma = generate_equicorr(p, rho)
        for i, n in enumerate(ns):
            for j in range(nsim):
                # Draw n points iid...
                loss_values = gaussian_ecdf_dist(n, p, tgrid, rho, Sigma = Sigma)
                Q[i, j], _ = bootstrap_UB(data = loss_values, 
                                        nboot = nboot, 
                                        delta = 0.1)
        quantiles10[str(rho)] = np.quantile(Q, 0.1, axis = 1)
        quantiles50[str(rho)] = np.quantile(Q, 0.5, axis = 1)
        quantiles90[str(rho)] = np.quantile(Q, 0.9, axis = 1)

    kwargs = {}
    for i, rho in enumerate(rhos):
        nm = str(np.round(rho, 1))
        kwargs['bootstrap_q10_r' + nm] = quantiles10[nm]
        kwargs['bootstrap_q50_r' + nm] = quantiles50[nm]
        kwargs['bootstrap_q90_r' + nm] = quantiles90[nm]


    np.savez('../results/gaussian_cdf_quantiles_bootstrap.npz', 
            ns = ns,
            **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    parser.add_argument("--ngrid", dest = 'ngrid', type = int, help="Number of grid points")
    parser.add_argument("--nboot", dest = 'nboot', type = int, help="Number of bootstrap samples")
    args = parser.parse_args()
    main(args)




