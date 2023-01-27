#!/usr/bin/env bash
DATASET_PATH="/tmp/nvflare/dataset/sklearn_iris.csv"

python3 utils/prepare_data.py \
--dataset_name iris \
--randomize 1 \
--out_path ${DATASET_PATH}

echo "Data loaded and saved in ${DATASET_PATH}"
