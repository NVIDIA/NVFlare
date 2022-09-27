#!/usr/bin/env bash
mkdir data_splits
for split_mode in uniform exponential
do
  python3 utils/prepare_data_split.py --data_path DATASET_PATH --site_num 5 --split_method ${split_mode} --out_path "data_splits/data_split_5_${split_mode}.json"
done

for split_mode in uniform square
do
  python3 utils/prepare_data_split.py --data_path DATASET_PATH --site_num 20 --split_method ${split_mode} --out_path "data_splits/data_split_20_${split_mode}.json"
done
