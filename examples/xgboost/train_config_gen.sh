#!/usr/bin/env bash
mkdir train_configs
for split_mode in uniform exponential
do
  python3 utils/prepare_train_config.py --site_num 5 --split_method ${split_mode} --out_path "train_configs/config_train_5_${split_mode}.json"
done

for split_mode in uniform square
do
  python3 utils/prepare_train_config.py --site_num 20 --nthread 4 --split_method ${split_mode} --out_path "train_configs/config_train_20_${split_mode}.json"
done
