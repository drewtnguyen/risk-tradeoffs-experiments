# Set global parameters 

NGRID=1000 # Grid spacing (on [-3, 3])
NBOOT=1000 # Number of bootstrap samples

# Set number of simulation runs for each experiment

NSIM_Q=10000   ## Quantiles
NSIM_B=3000    ## Bootstrap quantiles
NSIM_M=20000    ## Metrics

echo "computing quantiles..."
python sims_quantiles_gaussian_cdf.py --nsim $NSIM_Q --ngrid $NGRID
echo "computing bootstrap metrics of quantiles..."
python sims_bootstrap_gaussian_cdf.py --nsim $NSIM_B --ngrid $NGRID --nboot $NBOOT
echo "computing all other metrics..."
python sims_metrics_gaussian_cdf.py --nsim $NSIM_M --ngrid $NGRID --nboot $NBOOT
