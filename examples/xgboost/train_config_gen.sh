#!/usr/bin/env bash
for split_mode in uniform linear square exponential
do
  python3 prepare_train_config.py --split_method ${split_mode} --out_path "train_configs/config_train_${split_mode}.json"
done

