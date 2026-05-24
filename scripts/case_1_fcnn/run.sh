rm -r results
mkdir results

clear

python3 -u process_map_files.py
python3 -u main.py
python3 -u postprocess.py



