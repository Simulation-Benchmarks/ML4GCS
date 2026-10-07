# script to be run after run.sh

set -e

export PYTHONPATH=../../src

python3 postprocess.py
python3 plot_loss.py
python3 plot_r2.py
