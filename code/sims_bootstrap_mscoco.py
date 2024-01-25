import numpy as np
from util import subsample, bootstrap_UB
from math import floor
import argparse
from tqdm import tqdm


def main(args):
    nsim = args.nsim
    nboot = args.nboot

    # Load data
    mscoco_losses = np.load('../results/mscoco_losses.npz')
    ntotal = int(mscoco_losses['ntotal'])
    nholdout = ntotal // 2

    # Save the quantiles of the quantiles
    losstypes = ['fnp', 'fpp', 'fdp', 'ss']
    quantiles10 = {
        'fnp': None, 
        'fpp': None, 
        'fdp': None, 
        'ss': None
    }
    quantiles50 = {
        'fnp': None, 
        'fpp': None, 
        'fdp': None, 
        'ss': None
    }
    quantiles90 = {
        'fnp': None, 
        'fpp': None, 
        'fdp': None, 
        'ss': None
    }
    ns = np.array([50, 50*3, 50*(3**2), 50*(3**3)])
    # Each simulation run, set aside a holdout set of some size, 
    # draw n points iid from the remaining empirical distribution,
    # and compute a new bootstrap quantile. 
    total_metrics = len(losstypes) * len(ns)
    metrics_done = 0
    for lt in losstypes:
        loss = mscoco_losses[lt]
        Q = np.zeros([len(ns), nsim])
        for i, n in enumerate(ns):
            print("Computing metrics for settings " + str(metrics_done) + "/" + str(total_metrics) + ", n = " + str(n))
            metrics_done += 1
            for j in tqdm(range(nsim), leave = False):
                # Pull out the holdout set...
                idxh = np.array([1] * nholdout + [0] * (ntotal - nholdout)) > 0
                np.random.shuffle(idxh) 
                _, loss_ = loss[idxh, :], loss[~idxh, :]
                # Draw n points iid...
                loss_values = subsample(n, loss_)
                Q[i, j], _ = bootstrap_UB(data = loss_values, 
                                        nboot = nboot, 
                                        delta = 0.1)
        quantiles10[lt] = np.quantile(Q, 0.1, axis = 1)
        quantiles50[lt] = np.quantile(Q, 0.5, axis = 1)
        quantiles90[lt] = np.quantile(Q, 0.9, axis = 1)

    kwargs = {}
    for lt in losstypes:
            kwargs['bootstrap_q10_' + lt] = quantiles10[lt]
            kwargs['bootstrap_q50_' + lt] = quantiles50[lt]
            kwargs['bootstrap_q90_' + lt] = quantiles90[lt]

    np.savez('../results/mscoco_quantiles_bootstrap.npz', 
            ns = ns, 
            **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    parser.add_argument("--nboot", dest = 'nboot', type = int, help="Number of bootstrap samples")
    args = parser.parse_args()
    main(args)





