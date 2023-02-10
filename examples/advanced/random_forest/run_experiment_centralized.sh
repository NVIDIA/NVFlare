#!/usr/bin/env bash

python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.5 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.5 --split_option exponential --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.5 --split_option exponential --bagging_option weighted

python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.05 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.05 --split_option exponential --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.05 --split_option exponential --bagging_option weighted

python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.005 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.005 --split_option exponential --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 5 --subsample 0.005 --split_option exponential --bagging_option weighted

python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.8 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.8 --split_option square --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.8 --split_option square --bagging_option weighted

python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.2 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.2 --split_option square --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.2 --split_option square --bagging_option weighted

python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.02 --split_option uniform --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.02 --split_option square --bagging_option uniform
python3 baseline_bagging_RF.py --num_sites 20 --subsample 0.02 --split_option square --bagging_option weighted