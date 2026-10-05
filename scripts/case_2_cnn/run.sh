# script to fully run the test case

set -e

rm -rf results
mkdir results

clear

# python3 -u process_map_files.py
python3 -u main.py

python3 -u postprocess.py
python3 plot_loss.py


