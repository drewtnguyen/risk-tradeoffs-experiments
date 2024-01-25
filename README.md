# Experiments for ``Trading off multiple risks for predictive algorithms with confidence''


## Directory structure

The `code` directory contains all the code. All scripts must be run
from within that directory. 

The `data` directory is initially empty, but by following the steps 
in this README, is populated with sigmoid scores and targets
on 60K images from MS COCO (which constitute "Split 3" in Appendix A.2 of the 
paper).

The `results` directory is initially empty, but by following the steps in this README, is populated with simulation results that
are used for the figures. 

The `figures` directory contains all the figures of the paper, 
and can be re-generated based on the contents of `results`.

## Setup

First, create a virtual environment and install all dependencies: 

```
conda create --name risk-tradeoffs-project
conda activate risk-tradeoffs-project
pip install -r requirements.txt
```

and if not already in the code directory, navigate there:

```
cd code
```

Second, run the starter script:

```
sh run_first.sh
```

This downloads files related to the MS COCO experiments to the initially empty `data` directory. 

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
directory. 

## Generating figures

To generate all figures in the paper, from within the `code` directory, run

```
python generate_plots.py
```





