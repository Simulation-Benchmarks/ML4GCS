# script to fully run the test case

set -e

export PYTHONPATH=../../src

rm -rf results
mkdir results

clear

# run once: spe11b/ -> data/spe11b.h5
if [ ! -f ../../data/spe11b.h5 ]; then
    python3 -u -m spe11_wasserstein.preprocess
fi
python3 -u main.py

python3 -u postprocess.py
python3 plot_loss.py
python3 plot_r2.py


