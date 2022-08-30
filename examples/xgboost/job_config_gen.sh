#!/usr/bin/env bash
python3 utils/prepare_job_config.py --site_num 5 --training_mode bagging --split_method exponential --lr_mode scaled --nthread 16
python3 utils/prepare_job_config.py --site_num 5 --training_mode bagging --split_method exponential --lr_mode uniform --nthread 16
python3 utils/prepare_job_config.py --site_num 5 --training_mode bagging --split_method uniform --lr_mode uniform --nthread 16
python3 utils/prepare_job_config.py --site_num 5 --training_mode cyclic --split_method exponential --lr_mode uniform --nthread 16
python3 utils/prepare_job_config.py --site_num 5 --training_mode cyclic --split_method uniform --lr_mode uniform --nthread 16

python3 utils/prepare_job_config.py --site_num 20 --training_mode bagging --split_method square --lr_mode scaled --nthread 4
python3 utils/prepare_job_config.py --site_num 20 --training_mode bagging --split_method square --lr_mode uniform --nthread 4
python3 utils/prepare_job_config.py --site_num 20 --training_mode bagging --split_method uniform --lr_mode uniform --nthread 4
python3 utils/prepare_job_config.py --site_num 20 --training_mode cyclic --split_method square --lr_mode uniform --nthread 4
python3 utils/prepare_job_config.py --site_num 20 --training_mode cyclic --split_method uniform --lr_mode uniform --nthread 4
