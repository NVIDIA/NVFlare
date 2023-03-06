#!/bin/bash

DATASET_PATH="/tmp/nvflare/dataset/sklearn_breast_cancer.csv"
script_dir="$( dirname -- "$0"; )";

if [ -f "$DATASET_PATH" ]; then
    echo "${DATASET_PATH} exists."
else
    python3 ${script_dir}/utils/prepare_data.py \
        --dataset_name cancer \
        --randomize 0 \
        --out_path ${DATASET_PATH}
    echo "Data loaded and saved in ${DATASET_PATH}"
fi
