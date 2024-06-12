#!/usr/bin/env bash
DATASET_PATH="$HOME/dataset/HIGGS.csv"
OUTPUT_PATH="/tmp/nvflare/vertical_xgb_data"
OUTPUT_FILE="higgs.data.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

echo "Generating HIGGS data splits, reading from ${DATASET_PATH}"

python3 utils/prepare_data.py \
--data_path "${DATASET_PATH}" \
--site_num 2 \
--rows_total_percentage 0.02 \
--rows_overlap_percentage 0.25 \
--out_path "${OUTPUT_PATH}" \
--out_file "${OUTPUT_FILE}"

# Note: HIGGS has 11000000 preshuffled instances; using rows_total_percentage to reduce PSI time for example

echo "Data splits are generated in ${OUTPUT_PATH}"
