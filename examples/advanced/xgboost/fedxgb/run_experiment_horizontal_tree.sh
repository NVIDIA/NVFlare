#!/usr/bin/env bash
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo bagging --split_method exponential --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo bagging --split_method exponential --lr_mode scaled --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo bagging --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo cyclic --split_method exponential --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo cyclic --split_method uniform --lr_mode uniform --data_split_mode horizontal

python3 xgb_fl_job_horizontal.py --site_num 20 --training_algo bagging --split_method square --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 20 --training_algo bagging --split_method square --lr_mode scaled --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 20 --training_algo bagging --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 20 --training_algo cyclic --split_method square --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 20 --training_algo cyclic --split_method uniform --lr_mode uniform --data_split_mode horizontal