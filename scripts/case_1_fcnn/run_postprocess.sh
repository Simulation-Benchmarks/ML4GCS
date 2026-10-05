# script to be run after run.sh

set -e

python3 postprocess.py
python3 plot_loss.py
