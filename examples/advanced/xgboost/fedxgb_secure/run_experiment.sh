#!/usr/bin/env bash

# Secure Federated XGBoost Experiments using Recipe API
# This script demonstrates running both horizontal and vertical XGBoost
# with and without Homomorphic Encryption (HE)

dataset_path="/tmp/nvflare/dataset/xgb_dataset"

echo "====================================="
echo "Horizontal XGBoost (Non-Secure)"
echo "====================================="
python3 job.py --data_root "${dataset_path}/horizontal_xgb_data" --site_num 3 --round_num 3

echo ""
echo "====================================="
echo "Horizontal XGBoost (Secure with HE)"
echo "====================================="
echo "Note: This prepares the job but requires additional tenseal context setup"
echo "See README for instructions on running secure horizontal training"
python3 job.py --data_root "${dataset_path}/horizontal_xgb_data" --site_num 3 --round_num 3 --secure

echo ""
echo "====================================="
echo "Vertical XGBoost (Non-Secure)"
echo "====================================="
python3 job_vertical.py --data_root "${dataset_path}/vertical_xgb_data" --site_num 3 --round_num 3

echo ""
echo "====================================="
echo "Vertical XGBoost (Secure with HE)"
echo "====================================="
echo "Note: Requires encryption plugin to be installed and configured"
echo "Set NVFLARE_XGB_PLUGIN_NAME and NVFLARE_XGB_PLUGIN_PATH environment variables"
# NVFLARE_XGB_PLUGIN_NAME=nvflare NVFLARE_XGB_PLUGIN_PATH=/tmp/nvflare/plugins/libnvflare.so \
# python3 job_vertical.py --data_root "${dataset_path}/vertical_xgb_data" --site_num 3 --round_num 3 --secure
