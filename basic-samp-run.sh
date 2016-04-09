#
#  These generate the experiments that explore different sampling methods
#  on PPI-like synthetic graphs
#
python sampRun.py 500 0.02 GEO 
python sampRun.py 500 0.03 GEO 
python sampRun.py 500 0.06 GEO 
python sampRun.py 500 0.09 GEO 
python sampRun.py 500 0.0 SL 
cd samprun/noperturb
python ../../sampPlots.py

