#!/usr/bin/env bash
DATASET_PATH="$HOME/dataset/HIGGS.csv"
OUTPUT_PATH="/tmp/nvflare/vertical_xgboost/data"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

echo "Generating HIGGS data splits, reading from ${DATASET_PATH}"

python3 utils/prepare_data_split.py \
--data_path "${DATASET_PATH}" \
--site_num 2 \
--cols_total 29 \
--rows_overlap_size 50000 \
--rows_total 200000 \
--out_path "${OUTPUT_PATH}"

# Notes: HIGGS has 11000000 preshuffled instances; using subset to reduce PSI time for example

echo "Data splits are generated in ${OUTPUT_PATH}"
