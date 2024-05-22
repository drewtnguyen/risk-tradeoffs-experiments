# Experiments for "Data-Adaptive Tradeoffs among Multiple Risks in Distribution-Free Prediction"

This repository reproduces the figures in the paper "Data-Adaptive Tradeoffs among Multiple Risks in Distribution-Free Prediction" ([arxiv preprint here](https://arxiv.org/abs/2403.19605)). 

## Directory structure

The `code` directory contains all the code. All scripts must be run
from within that directory. 

The `data` directory initially only has one file, `coco_examples_indices.npy`, but by following the steps 
in this README, is populated with sigmoid scores and targets
on 60K images from MS COCO (which constitute "Split 3" in Appendix A.2 of the 
paper).

The `results` directory is initially empty, but by following the steps in this README, is populated with simulation results that
are used for the figures. 

The `figures` directory contains all the figures of the paper, 
and can be re-generated based on the contents of `results`.

## Setup

First, clone the repo, navigate to it, and create a virtual environment and install all dependencies:

```
git clone git@github.com:drewtnguyen/risk-tradeoffs-experiments.git
cd risk-tradeoffs-experiments
conda create --name risk-tradeoffs-experiments
conda activate risk-tradeoffs-experiments
pip3 install -r requirements.txt
```

Second, navigate to the `code` directory, and run the starter script:

```
cd code
sh run_first.sh
```

This downloads files related to the MS COCO experiments to the `data` directory. 

## Running experiments

First, if not already in the `code` directory, navigate there:

```
cd code
```

To replicate the experiments of Section 3.1, run

```
sh run_experiments_gaussian_cdf.sh
```

To replicate the experiments of the Introduction and Section 3.2, run

```
sh run_experiments_mscoco.sh
```
These scripts populate the `results`
directory. (The number of simulation runs, and other parameters, can be changed.)

## Generating figures


To generate all figures in the paper, from within the `code` directory, run

```
python3 generate_plots.py
```





