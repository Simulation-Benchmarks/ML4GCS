# script to be run after run.sh

set -e

cd "$(dirname "$0")"

python3 postprocess.py
python3 plot_loss.py
