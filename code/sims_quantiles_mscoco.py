import numpy as np
from util import compute_UB_qtl_holdout
from math import floor
import argparse

def main(args):
    nsim = args.nsim

    # Load data
    mscoco_losses = np.load('../results/mscoco_losses.npz')
    ntotal = int(mscoco_losses['ntotal'])
    nholdout = ntotal // 2

    # Save the quantiles 
    losstypes = ['fnp', 'fpp', 'fdp', 'ss']
    quantiles = {
        'fnp': None, 
        'fpp': None, 
        'fdp': None, 
        'ss': None
    }
    ns = np.array([50, 50*3, 50*(3**2), 50*(3**3)])

    for lt in losstypes:
        loss_values = mscoco_losses[lt]
        # Compute KS statistic quantiles for various values of n
        # maginalizing over choice of truvec.
        # But n should not be taken too large.
        Q = np.zeros(len(ns))
        for i, n in enumerate(ns):
            Q[i], _ = compute_UB_qtl_holdout(n = n, 
                    nholdout = nholdout, 
                    datadist = loss_values, 
                    delta = 0.1, 
                    reps = nsim)
        quantiles[lt] = Q

    # The values of the quantiles increase in n, presumably due to the error in the holdout set.
    np.savez('../results/mscoco_quantiles.npz', 
            ns = ns, 
            fnp = quantiles['fnp'], 
            fpp = quantiles['fpp'],
            fdp = quantiles['fdp'], 
            ss = quantiles['ss'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    args = parser.parse_args()
    main(args)


