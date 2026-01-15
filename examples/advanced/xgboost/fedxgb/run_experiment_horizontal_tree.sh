#!/usr/bin/env bash
# Tree-based federated XGBoost experiments (bagging and cyclic modes)
# Now using Recipe API via job_tree.py

echo "Running 5-client experiments..."
python3 job_tree.py --site_num 5 --training_algo bagging --split_method exponential --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 5 --training_algo bagging --split_method exponential --lr_mode scaled --data_split_mode horizontal
python3 job_tree.py --site_num 5 --training_algo bagging --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 5 --training_algo cyclic --split_method exponential --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 5 --training_algo cyclic --split_method uniform --lr_mode uniform --data_split_mode horizontal

echo "Running 20-client experiments..."
python3 job_tree.py --site_num 20 --training_algo bagging --split_method square --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 20 --training_algo bagging --split_method square --lr_mode scaled --data_split_mode horizontal
python3 job_tree.py --site_num 20 --training_algo bagging --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 20 --training_algo cyclic --split_method square --lr_mode uniform --data_split_mode horizontal
python3 job_tree.py --site_num 20 --training_algo cyclic --split_method uniform --lr_mode uniform --data_split_mode horizontal
