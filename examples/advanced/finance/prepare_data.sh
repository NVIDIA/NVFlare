#!/usr/bin/env bash
DATASET_PATH="${PWD}/dataset/creditcard.csv"
SPLIT_PATH="${PWD}/dataset"

OUTPUT_PATH_HOR="/tmp/dataset/horizontal_xgb_data"

OUTPUT_PATH_VER="/tmp/dataset/vertical_xgb_data"
OUTPUT_FILE_VER="data.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved CreditCard dataset in ${DATASET_PATH}"
fi

echo "Generating CreditCard data splits, reading from ${DATASET_PATH}"

python3 utils/prepare_data_split.py \
--data_path "${DATASET_PATH}" \
--test_ratio 0.2 \
--out_folder "${SPLIT_PATH}"

python3 utils/prepare_data_horizontal.py \
--data_path "${SPLIT_PATH}/train.csv" \
--site_num 2 \
--split_method "uniform" \
--out_path "${OUTPUT_PATH_HOR}"

python3 utils/prepare_data_vertical.py \
--data_path "${SPLIT_PATH}/train.csv" \
--site_num 2 \
--out_path "${OUTPUT_PATH_VER}" \
--out_file "${OUTPUT_FILE_VER}"
