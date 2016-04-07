# matching-metrics

Tools for experiments on metrics used for matching.

apxgi.py runs MCMC to estimate the distribution of node correctness for a given edge correctness.

# experiments

To generate results for synthetic graphs:

python run.py 500 0.03 ER noperturb
python run.py 500 0.06 ER noperturb
python run.py 500 0.09 ER noperturb
python run.py 500 0.004 BA noperturb
python run.py 500 0.01 BA noperturb
python run.py 500 0.03 BA noperturb
python run.py 500 0.06 BA noperturb
python run.py 500 0.09 BA noperturb
python run.py 500 0.0001 WS noperturb
python run.py 500 0.001 WS noperturb
python run.py 500 0.01 WS noperturb
python run.py 500 0.05 WS noperturb
python run.py 500 0.02 GEO noperturb
python run.py 500 0.03 GEO noperturb
python run.py 500 0.06 GEO noperturb
python run.py 500 0.09 GEO noperturb
python run.py 500 0.1 EV noperturb
python run.py 500 0.24 EV noperturb
python run.py 500 0.0 SL noperturb

To generate perturbed results:

python run.py 500 0.03 ER thin 0.25
(etc for scramble, rewire, randomize)

To generate results comparing different graph sampling methods:

python sampRun.py 500 0.02 GEO 
python sampRun.py 500 0.03 GEO 
python sampRun.py 500 0.06 GEO 
python sampRun.py 500 0.09 GEO 
python sampRun.py 500 0.0 SL 
cd samprun/noperturb
python ../../sampPlots.py

To generate results using XS sampling on PPI graphs:

python run.py 500 0 PPI noperturb 0 human 
or
python run.py 500 0 PPI thin 0.25 human
