import numpy as np
import os
import loss_functions as loss
from sklearn.metrics import average_precision_score
import argparse

def main(args):
    ngrid = args.ngrid

    # Select best performing model on a val set
    np.random.seed(1)
    mAPs = []
    nval = 1000
    ntotal = 58633 # size of datasets
    datapath = '../data/coco_epochs/'
    idxv = np.array([1] * nval + [0] * (ntotal - nval)) > 0
    np.random.shuffle(idxv) 
    for i in range(29):
        epoch = str(i + 1)
        pfn = datapath + 'preds-' + epoch + '-1833.npy'
        tfn = datapath + 'targets-' + epoch + '-1833.npy'
        preds = np.load(pfn)
        targets = np.load(tfn)
        predsval = preds[idxv, :]
        targetsval = targets[idxv, :]
        mAP = average_precision_score(targetsval, predsval)
        mAPs.append(mAP)
    mAPs = np.asarray(mAPs)
    best_epoch = str(np.argmax(mAPs) + 1)
    print("Best epoch was " + best_epoch)

    # Now load data for best model
    pfn = datapath + 'preds-' + best_epoch + '-1833.npy'
    tfn = datapath + 'targets-' + best_epoch + '-1833.npy'
    sgmd = np.load(pfn)
    labels = np.load(tfn)
    # Remove the validation set
    sgmd = sgmd[~idxv, :]
    labels = labels[~idxv, :]

    # Generate losses on grid
    tgrid = np.linspace(0, 1, ngrid)
    fnp_vals = loss.FNP(tgrid, sgmd, labels)
    fpp_vals = loss.FPP(tgrid, sgmd, labels)
    fdp_vals = loss.FDP(tgrid, sgmd, labels)
    ss_vals = loss.SetSize(tgrid, sgmd, labels)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        
    ntotal = sgmd.shape[0]

    np.savez('../results/mscoco_losses.npz', ntotal = ntotal, tgrid = tgrid, fnp = fnp_vals, fpp = fpp_vals, fdp = fdp_vals, ss = ss_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrid", dest = 'ngrid', type = int, help="Number of grid points")
    args = parser.parse_args()
    main(args)

