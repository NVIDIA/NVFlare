#!/usr/bin/env bash
DATASET_PATH="/tmp/nvflare/dataset/sklearn_breast_cancer.csv"

python3 utils/prepare_data.py \
--dataset_name cancer \
--randomize 0 \
--out_path ${DATASET_PATH}

echo "Data loaded and saved in ${DATASET_PATH}"
