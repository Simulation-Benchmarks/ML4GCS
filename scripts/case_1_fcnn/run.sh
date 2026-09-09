# script to fully run the test case

set -e

cd "$(dirname "$0")"

rm -rf results
mkdir results

clear

python3 -u ../prepare_wasserstein_data.py --output-dir .
python3 -u main.py

python3 -u postprocess.py
python3 plot_loss.py
