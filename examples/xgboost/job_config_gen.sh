#!/usr/bin/env bash
python3 utils/prepare_job_config.py --site_num 5 --train_mode bagging --split_method exponential --lr_mode scaled
python3 utils/prepare_job_config.py --site_num 5 --train_mode bagging --split_method exponential --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 5 --train_mode bagging --split_method uniform --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 5 --train_mode cyclic --split_method exponential --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 5 --train_mode cyclic --split_method uniform --lr_mode uniform

python3 utils/prepare_job_config.py --site_num 20 --train_mode bagging --split_method square --lr_mode scaled
python3 utils/prepare_job_config.py --site_num 20 --train_mode bagging --split_method square --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 20 --train_mode bagging --split_method uniform --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 20 --train_mode cyclic --split_method square --lr_mode uniform
python3 utils/prepare_job_config.py --site_num 20 --train_mode cyclic --split_method uniform --lr_mode uniform