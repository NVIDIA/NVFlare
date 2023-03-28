#!/usr/bin/env bash
DATASET_ROOT=${1}
python3 utils/data_split.py --data_path ${DATASET_ROOT} --num_clients 4 --random_seed 0 --site_name_prefix 'site-'
