#!/usr/bin/env bash
python3 xgb_fl_job_horizontal.py --site_num 2 --training_algo histogram --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo histogram --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 2 --training_algo histogram_v2 --split_method uniform --lr_mode uniform --data_split_mode horizontal
python3 xgb_fl_job_horizontal.py --site_num 5 --training_algo histogram_v2 --split_method uniform --lr_mode uniform --data_split_mode horizontal
