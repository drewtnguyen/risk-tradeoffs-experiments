# Set global parameters 

NGRID=500 # Grid spacing (on [0, 1])
NBOOT=1000 # Number of bootstrap samples

# Set number of simulation runs for each experiment

NSIM_Q=10000   ## Quantiles
NSIM_B=3000    ## Bootstrap quantiles
NSIM_M=20000    ## Metrics

# First compute the losses (on a grid)
echo "computing losses..."
python3 setup_compute_losses_mscoco.py --ngrid $NGRID
# Then run the simulations
echo "computing quantiles..."
python3 sims_quantiles_mscoco.py --nsim $NSIM_Q 
echo "computing bootstrap metrics of quantiles..."
python3 sims_bootstrap_mscoco.py --nsim $NSIM_B  --nboot $NBOOT
echo "computing all other metrics..."
python3 sims_metrics_mscoco.py --nsim $NSIM_M --nboot $NBOOT



