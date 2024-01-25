# Compute coverage and conservatism for mscoco
import numpy as np
from util import compute_UB_metrics_holdout
from loss_functions import monotonize_decr
import argparse

def main(args):
    nsim = args.nsim
    nboot = args.nboot

    # Load data
    mscoco_losses = np.load('../results/mscoco_losses.npz')
    ntotal = int(mscoco_losses['ntotal'])
    nholdout = ntotal // 2

    losstypes = ['fnp', 'fpp', 'fdp', 'ss']
    losstypes_tr = ['fpp', 'fnp', 'fnp', 'fnp']

    # ns = np.array([100, 150, 200, 250, 500, 50, 50*3, 50*(3**2), 50*(3**3)])
    ns = np.array([100, 300, 500, 50, 50*3, 50*(3**2), 50*(3**3)])
    nreps = nsim

    miscovers_by_loss = {
        'fnp': [],
        'fpp': [],
        'fdp': [],
        'ss': []
    }

    conservatism_by_loss = {
        'fnp': [],
        'fpp': [],
        'fdp': [],
        'ss': []
    }

    pmiscovers_by_loss = {
        'fnp': [],
        'fpp': [],
        'fdp': [],
        'ss': []
    }

    tmiscovers_by_loss = {
        'fnp': [],
        'fpp': [],
        'fdp': [],
        'ss': []
    }


    def monotonize_fdp(loss, lt):
        if lt == 'fdp':
            loss = monotonize_decr(loss)
        return loss

    total_metrics = len(losstypes) * len(ns)
    metrics_done = 0
    for lt, ltr in zip(losstypes, losstypes_tr):
        for i, n in enumerate(ns):
            loss = mscoco_losses[lt]
            loss_tr = mscoco_losses[ltr]
            loss = monotonize_fdp(loss, lt)
            loss_tr = monotonize_fdp(loss_tr, ltr)
            if lt in {'fnp', 'fpp'}:
                tradeoff_type = 'equal_weighted_sum'
            else:
                tradeoff_type = 'elbow'
            print("Computing metrics for settings " + str(metrics_done) + "/" + str(total_metrics) + ", n = " + str(n))
            metrics_done += 1
            miscover, conservatism = compute_UB_metrics_holdout(
                n = n, nholdout = nholdout, nboot = nboot,
                datadist = loss,
                tradeoff_datadist = loss_tr, 
                tradeoff_type = tradeoff_type,
                alphamax = 0.1,
                delta = 0.1, 
                reps = nreps
                )
            miscovers_by_loss[lt].append(miscover)
            conservatism_by_loss[lt].append(conservatism)


    types = [
            'bootstrap_anywhere',
            'bootstrap_tradeoff',
            'bootstrap_tradeoff_alphamax',
            'bootstrap_selected',
            'DKW_anywhere',
            'DKW_tradeoff',
            'DKW_tradeoff_alphamax',
            'DKW_selected',
            'locsim_anywhere',
            'locsim_tradeoff',
            'locsim_tradeoff_alphamax',
            'locsim_selected',
            'pointwise_anywhere',
            'pointwise_tradeoff',
            'pointwise_tradeoff_alphamax',
            'pointwise_selected'
    ]



    mkwargs = {}
    ckwargs = {}

    for lt in losstypes:
        for i, n in enumerate(ns):
            for t in types:
                mkwargs[lt + '_' + t + '_' + str(n)] = miscovers_by_loss[lt][i][t]
                ckwargs[lt + '_' + t + '_' + str(n)] = conservatism_by_loss[lt][i][t]

    mkwargs['ns'] = ns
    ckwargs['ns'] = ns

    np.savez('../results/mscoco_miscover.npz', **mkwargs)
    np.savez('../results/mscoco_conservatism.npz', **ckwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", dest = 'nsim', type = int, help="Number of simulation runs")
    parser.add_argument("--nboot", dest = 'nboot', type = int, help="Number of bootstrap samples")
    args = parser.parse_args()
    main(args)

