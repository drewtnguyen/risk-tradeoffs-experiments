import numpy as np
import matplotlib.pyplot as plt
from math import e
import seaborn as sns
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

################################
# Define plotting helpers
#################################


def quantiles_plot(fn, ns, xlabels, 
                    Q, Qb10, Qb50, Qb90):
    fig, ax = plt.subplots(1, len(xlabels), sharex = True, sharey = True)
    for i, xlabel in enumerate(xlabels):
        DKW = np.sqrt(np.log(e/0.1) / (2*ns))
        ax[i].loglog(ns, Q[i], linestyle = '--', color = 'green', label = 'True Quantile')
        ax[i].loglog(ns, Qb50[i], linestyle = '-', marker = 'o', color = 'blue', alpha = 0.85, label = 'Bootstrap est.')
        ax[i].loglog(ns, DKW, linestyle = '--', color = 'gray', label = 'DKW')
        ax[i].fill_between(ns, Qb10[i], Qb90[i], color='lightblue', alpha=0.5)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_xlabel(xlabel)
    fig.set_figwidth(16)
    fig.set_figheight(4)
    # fig.set_figwidth(4)
    # fig.set_figwidth(6)
    ax[0].set_ylabel(r'1 - $\delta$ Quantile of $D_n$')
    ax[-1].legend(loc = 'lower right')
    plt.savefig(fn,bbox_inches='tight')


def metrics_barplot(fn, metricsnpz, ylabel, 
                    ns, losstypes, losstypes2, algorithms, 
                    alglabels, where):
    fig, ax = plt.subplots(1, len(losstypes), sharex = True, sharey = True)
    # Set the width of the bars
    bar_width = 0.2
    algindex = np.arange(len(algorithms))
    for i, lt in enumerate(losstypes):
        dataarray = np.zeros((len(algorithms), len(ns)))
        for ii, alg in enumerate(algorithms):
            for jj, n in enumerate(ns):
                key = '_'.join([lt, alg, where, str(n)])
                dataarray[ii,jj] = metricsnpz[key].mean()
        for jj, n in enumerate(ns):
            label = 'n = ' + str(ns[jj])
            ax[i].bar(algindex + jj * bar_width, dataarray[:, jj], bar_width, label=label)
        ax[i].set_xlabel(losstypes2[i])
        ax[i].set_xticks(algindex + 1.5 * bar_width)
        ax[i].set_xticklabels(alglabels)
        ax[i].axhline(0.1, color = 'gray', linestyle='--')
    ax[0].set_ylabel(ylabel)
    ax[-1].legend()
    fig.set_figwidth(15)
    plt.savefig(fn, bbox_inches='tight')



def plot_histogram(fn, bins, metricsnpz, 
                    ns, algorithms, alglabels, loss, delta = 0.1, 
                    xlim_right = None, xlim_left = None, 
                    ylim_top = None, ylim_bottom = None, 
                    showmeans = True, 
                    alphas = [0.2, 0.6]):
    # Set up a panel for each n
    fig, ax = plt.subplots(nrows=1, ncols=len(ns), sharey = True, sharex = True)    
    # Plot horizontally oriented histograms for each method, within each panel
    for ii, alg in enumerate(algorithms):
        for jj, n in enumerate(ns):
            key = '_'.join([loss, alg, 'tradeoff', str(n)])
            data = metricsnpz[key]
            # Plot the histogram
            ax[jj].hist(data, bins = bins, orientation = 'horizontal', 
                alpha = alphas[ii], density = True, label=alglabels[ii])
            ax[jj].set_xlim(right = xlim_right, left = xlim_left) 
            ax[jj].set_ylim(bottom = ylim_bottom, top = ylim_top) 
            # Make y-ticks visible
            ax[jj].yaxis.set_tick_params(labelbottom=True)
            # Add lines for the means
            if showmeans:
                histcolor = 'C' + str(ii)
                mean_val = np.mean(data)
                # Plot them on top of the axes
                ax[jj].plot(0, mean_val, color = histcolor, marker = 'x', 
                    markersize = 7, mew=2, zorder=10, clip_on=False, alpha = 0.8)
            if alg == 'pointwise':
                # we'll color the outer edge of the 10% quantile with a step plot
                # define the original histogram edges
                h, edges = np.histogram(data, bins=bins, density = True)
                # get new edges based on the quantile
                qtl = np.quantile(data, delta)
                goodh = []; goodedges = []; badh = []; badedges = []
                isgood = True
                for i, edge in enumerate(edges):
                    if isgood:
                        if i > 0:
                            goodh.append(h[i - 1])
                        if edge < qtl:
                            goodedges.append(edge)
                        else:
                            goodedges.append(qtl)
                            badedges.append(qtl)
                            badh.append(h[i - 1])
                            isgood = False
                            if edge < 0:
                                badedges.append(edge)
                            else:
                                badedges.append(0)
                                break
                    else:
                        badh.append(h[i - 1])
                        if edge < 0:
                            badedges.append(edge)
                        else:
                            badedges.append(0)
                            break
                ax[jj].stairs(goodh, goodedges, orientation = 'horizontal', color = 'green', 
                    linewidth = 1.5)
                if qtl < 0:
                    pass
                    # ax[jj].stairs(badh, badedges, orientation = 'horizontal', color = 'red', 
                    #     linewidth = 1.5)
    current_xticks = ax[0].get_xticks()
    new_xticks = [current_xticks[0], (current_xticks[-2] + current_xticks[-1])/2]
    # For each panel,
    for jj, n in enumerate(ns):
        # Add a vertical line at zero
        ax[jj].axhline(0, color = '#888888', linestyle='--')
        # Set labels, 
        ax[jj].set_xlabel('n = ' + str(ns[jj]))
        # ticks,
        ax[jj].set_xticks(new_xticks)
        ax[jj].tick_params(bottom = False) 
        # remove spines
        ax[jj].spines['right'].set_visible(False)
        ax[jj].spines['top'].set_visible(False)
    # Add a label to the left most plot
    ax[0].set_ylabel('risk gap')
    # Add a legend to the rightmost plot (in reversed order)
    handles, labels = ax[-1].get_legend_handles_labels()
    ax[-1].legend(handles[::-1], labels[::-1])
    # Resize and save
    fig.set_figwidth(15)
    fig.set_figheight(4)
    plt.savefig(fn, bbox_inches='tight')






################################
# Miscoverage with pointwise bounds
#################################

# mscoco_losses = np.load('../losses/mscoco_losses.npz')
# tgrid = mscoco_losses['tgrid']
# pmisc = np.load('../losses/mscoco_pmiscover.npz')
# ns = [100, 150, 200, 250, 500, 1350]

# losses = ['fnp', 'fdp']
# for loss in losses:
#     fixed_t_probs = [pmisc[loss + '_ppointwise_' + str(n)] for n in ns]
#     tradeoff_t_prob = [pmisc[loss + '_ptradeoff_' + str(n)] for n in ns]
#     fn = '../figures/' + loss + '_pointwise_miscoverage.pdf'
#     pointwise_miscoverage_plot(fn, tgrid, ns, fixed_t_probs, tradeoff_t_prob)

################################
# Teaser histogram
#################################

metricsnpz = np.load('../results/mscoco_conservatism.npz')
fn = '../figures/teaser_histogram.pdf'
bins = 80
ns = [100, 300, 500]
algorithms = ['locsim', 'pointwise']
alglabels = [r"RRR Method ($\delta=0.1$)", r'Previous Method ($\delta=0.1$)']
loss = 'fnp'
plot_histogram(fn, bins, metricsnpz, ns, algorithms, alglabels, loss, 
    ylim_top = 0.035, ylim_bottom = -0.02, xlim_right = 60, showmeans = False)

################################
# MS-COCO risk plots: Intro
#################################

mscoco_losses = np.load('../results/mscoco_losses.npz')
losstypes = ['fnp', 'fpp']
risktypes = {'fnp': 'FNR', 'fpp': 'FPR'}
t = mscoco_losses['tgrid']
fig, axs = plt.subplots(nrows = 1, ncols = 2)
fn = '../figures/mscoco_risks.pdf'
humanlabellist = np.load('../data/coco_human_readable_labels.npy')

# Get images that (probably) match something in the example set
imgdir = '../data/coco_image_examples/'
example_idx = np.load('../mscoco_examples_indices.npy', allow_pickle = True)
imglist = os.listdir(imgdir)
imgints = [int(os.path.splitext(img)[0]) for img in imglist]
imgints.sort()
imglist = [imgdir + str(imgint) + '.jpg' for imgint in imgints]
imgfn_idxs = [i for i, idx in enumerate(example_idx) if idx is not None]

# PLot the main curves
for lt in losstypes:
    loss = mscoco_losses[lt]
    truvec = loss.mean(axis = 0)
    axs[0].plot(1 - t, truvec, label = risktypes[lt])


thats_path = '../results/t_hats_figure.npz'

if os.path.isfile(thats_path):
    thatsz = np.load(thats_path)
    thats = thatsz['thats']
else:
    nreps = 300 
    n = 500
    thats = np.zeros(nreps)
    for i in range(nreps):
        print(i)
        fnp = mscoco_losses['fnp']
        fpp = mscoco_losses['fpp']
        nrow = fnp.shape[0]
        idx = np.random.choice(nrow, size = n, replace = True)
        objective = fnp[idx,].mean(axis = 0) + fpp[idx,].mean(axis = 0)
        istar = np.argmin(objective)
        tstar = t[istar]
        thats[i] = tstar
    np.savez('../results/t_hats_figure.npz', thats = thats)

# Add jitter and plot rug
thats = np.random.normal(thats,0.005)
sns.rugplot(1 - thats, linewidth = 0.2, height=0.025, ax=axs[0], color='k', alpha = 0.4)
sns.despine()


# Add image annotations

#ridx = np.random.choice(len(imgfn_idxs))
ridx = 46
imgidx = imgfn_idxs[ridx]
dataidx = example_idx[imgidx]
imfn = imglist[imgidx]
print(ridx)
print(imfn)
img = plt.imread(imfn)
axs[1].axis('off')
axs[1].imshow(img, aspect='equal')
# Get the image ground truth
sgmd_long = np.load('../data/coco_epochs/preds-5-1833.npy')
labels_long = np.load('../data/coco_epochs/targets-5-1833.npy')
n_long = sgmd_long.shape[0]

# Text annotations: 
# Pick t values
adj = 0.15
tsmall = 0.2 + adj
tmean = thats.mean() + adj
tbig = 0.7 + adj

label_img = labels_long[dataidx] != 0
humanlabel = str.join('\n',humanlabellist[label_img]) # Ground truth

sgmd_img = sgmd_long[dataidx]
labelsmall = sgmd_img > tsmall
labelmean = sgmd_img > tmean
labelbig = sgmd_img > tbig

hlabelsmall = str.join('\n',humanlabellist[labelsmall])
hlabelmean = str.join('\n',humanlabellist[labelmean])
hlabelbig = str.join('\n',humanlabellist[labelbig])


fontsize = 7

props = dict(boxstyle='round', facecolor='white', alpha=0.6)
axs[0].annotate(text = hlabelsmall, xy = (1 - tsmall, 0), xytext = (1 - tsmall,0.2),transform=axs[0].transAxes,fontsize=fontsize,color='k',horizontalalignment='center',verticalalignment='bottom',bbox=props, arrowprops = dict(arrowstyle = '-', facecolor='black'))
axs[0].annotate(text = hlabelmean, xy = (1 - tmean, 0), xytext = (1 - tmean,0.45),transform=axs[0].transAxes,fontsize=fontsize,color='k',horizontalalignment='center',verticalalignment='bottom',bbox=props, arrowprops = dict(arrowstyle = '-', facecolor='black'))
axs[0].annotate(text = hlabelbig, xy = (1 - tbig, 0), xytext = (1 - tbig,0.6),transform=axs[0].transAxes,fontsize=fontsize,color='k',horizontalalignment='center',verticalalignment='bottom',bbox=props, arrowprops = dict(arrowstyle = '-', facecolor='black'))

axs[1].text(0.05,0.95,humanlabel,transform=axs[1].transAxes,fontsize=10,color='k',verticalalignment='top',bbox=props)

axs[0].legend(loc = 'upper right')
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
axs[0].set_xlabel('threshold')  # Set x-axis label
axs[0].set_ylabel('risk')  # Set y-axis label
fig.set_figwidth(15)
fig.set_figheight(4)
plt.savefig(fn, bbox_inches='tight')



################################
# MS-COCO risk plots: Appendix
#################################

n = 300

mscoco_losses = np.load('../results/mscoco_losses.npz')
losstypes = ['fnp', 'fpp', 'fdp', 'ss']
risktypes = {'fnp': 'FNR', 'fpp': 'FPR', 'fdp' : 'FDP', 'ss' : 'SetSize'}
t = mscoco_losses['tgrid']
fig, axs = plt.subplots(nrows = 1, ncols = 4, sharex = True, sharey = True)
fn = '../figures/mscoco_risks_appendix.pdf'

# PLot the main curves
for i, lt in enumerate(losstypes):
    loss = mscoco_losses[lt]
    hidx = np.random.choice(loss.shape[0], size = loss.shape[0] // 2, replace = False)
    hloss, sloss = loss[hidx, ], loss[~hidx, ]
    sidx = np.random.choice(sloss.shape[0], size = n, replace = True)
    losssamp = sloss[sidx,]
    truvec = hloss.mean(axis = 0)
    empvec = losssamp.mean(axis = 0)
    axs[i].plot(1 - t, empvec, label = 'Empir.', 
                alpha = 0.7, color = 'C1')
    axs[i].plot(1 - t, truvec, label = risktypes[lt], color = 'C0')
    axs[i].set_xlabel('threshold')
    # Change order of legend https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = axs[i].get_legend_handles_labels()
    order = [1,0]
    axs[i].legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # Spines
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

axs[0].set_ylabel('risk')
fig.set_figwidth(15)
fig.set_figheight(4)
plt.savefig(fn, bbox_inches='tight')



################################
# Quantile plots
#################################

# Gaussian    

# gaussian_bs = np.load('../results/gaussian_cdf_bootstrap.npz')
# gaussian_qs = np.load('../results/gaussian_cdf_quantiles.npz')
# ns = gaussian_qs['ns']

# rhos = [-0.2, 0.2, 0.6]
# Q = [gaussian_qs['dependence_' + str(rho)] / np.sqrt(ns) for rho in rhos]
# Qb10 = [gaussian_bs['bootstrap_q10_r' + str(rho)] / np.sqrt(ns) for rho in rhos]
# Qb50 = [gaussian_bs['bootstrap_q50_r' + str(rho)] / np.sqrt(ns) for rho in rhos]
# Qb90 = [gaussian_bs['bootstrap_q90_r' + str(rho)] / np.sqrt(ns) for rho in rhos]
# xlabels = [r'$\rho$ = ' + str(rho) for rho in rhos]
# fn = '../figures/gaussian_quantiles.pdf'
# quantiles_plot(fn, ns, xlabels, Q, Qb10, Qb50, Qb90)

# MS-COCO

mscoco_bs = np.load('../results/mscoco_quantiles_bootstrap.npz')
mscoco_qs = np.load('../results/mscoco_quantiles.npz')
ns = mscoco_qs['ns']

losstypes = ['fnp', 'fpp', 'fdp', 'ss']
Q = [mscoco_qs[lt] / np.sqrt(ns) for lt in losstypes]
Qb10 = [mscoco_bs['bootstrap_q10_' + lt] / np.sqrt(ns) for lt in losstypes]
Qb50 = [mscoco_bs['bootstrap_q50_' + lt] / np.sqrt(ns) for lt in losstypes]
Qb90 = [mscoco_bs['bootstrap_q90_' + lt] / np.sqrt(ns) for lt in losstypes]
xlabels = ['FNP', 'FPP', 'FDP', 'SetSize']
fn = '../figures/mscoco_quantiles.pdf'
quantiles_plot(fn, ns, xlabels, Q, Qb10, Qb50, Qb90)


################################
# Miscoverage plots
#################################

# Gaussians
# losstypes = ['dependence_-0.2', 'dependence_0.2', 'dependence_0.6']
# losstypes2 = [r'$\rho$ = -0.2', r'$\rho$ = 0.2', r'$\rho$ = 0.6']

# algorithms = ['DKW', 'bootstrap', 'locsim', 'pointwise']
# ns = [30,  300, 3000]

# metricsnpz = np.load('../losses/gaussian_cdf_miscover.npz')
# where = 'anywhere'
# fn =  '../figures/gaussian_anywhere_miscoverage.pdf'
# ylabel = 'Anywhere Miscoverage'
# metrics_barplot(fn, metricsnpz, ylabel, ns, losstypes, losstypes2, algorithms, where)


# where = 'selected'
# fn =  '../figures/gaussian_selected_miscoverage.pdf'
# ylabel = 'Selected Set Miscoverage'
# metrics_barplot(fn, metricsnpz, ylabel, ns, losstypes, losstypes2, algorithms, where)


# Mscoco
losstypes = ['fnp', 'fpp', 'fdp', 'ss']
losstypes2 = ['FNR', 'FPR', 'FDR', 'SetSize']
algorithms = ['DKW', 'bootstrap', 'locsim', 'pointwise']
alglabels = ['BDKW', 'RR', 'RRR', 'pointwise']
ns = [50,  150,  450, 1350]



metricsnpz = np.load('../results/mscoco_miscover.npz')
where = 'anywhere'
fn =  '../figures/mscoco_anywhere_miscoverage.pdf'
ylabel = 'Anywhere Miscoverage'
metrics_barplot(fn, metricsnpz, ylabel, ns, losstypes, losstypes2, algorithms, alglabels, where)
where = 'selected'
ylabel = 'Selected Set Miscoverage'
fn =  '../figures/mscoco_selected_miscoverage.pdf'
metrics_barplot(fn, metricsnpz, ylabel, ns, losstypes, losstypes2, algorithms, alglabels, where)


metricsnpz = np.load('../results/mscoco_conservatism.npz')
where = 'tradeoff'
fn =  '../figures/mscoco_tradeoff_conservatism.pdf'
ylabel = 'Tradeoff Conservatism'
metrics_barplot(fn, metricsnpz, ylabel, ns, losstypes, losstypes2, algorithms, alglabels, where)
