# script to fully run the test case

set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

rm -rf results
mkdir results

clear

# python3 -u process_map_files.py
python3 -u main.py

python3 -u postprocess.py
python3 plot_loss.py


