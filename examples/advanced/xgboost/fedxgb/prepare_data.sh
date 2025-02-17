#!/usr/bin/env bash
DATASET_PATH="${1}/HIGGS.csv"
if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
    exit 1
fi

echo "Generating HIGGS data splits, reading from ${DATASET_PATH}"

OUTPUT_PATH="/tmp/nvflare/dataset/xgboost_higgs_horizontal"
for site_num in 2 5 20;
do
    for split_mode in uniform exponential square;
    do
        python3 utils/prepare_data_horizontal.py \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --size_total 11000000 \
        --size_valid 1000000 \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Horizontal data splits are generated in ${OUTPUT_PATH}"

OUTPUT_PATH="/tmp/nvflare/dataset/xgboost_higgs_vertical"
OUTPUT_FILE="higgs.data.csv"
# Note: HIGGS has 11 million preshuffled instances; using rows_total_percentage to reduce PSI time for example
python3 utils/prepare_data_vertical.py \
--data_path "${DATASET_PATH}" \
--site_num 2 \
--rows_total_percentage 0.02 \
--rows_overlap_percentage 0.25 \
--out_path "${OUTPUT_PATH}" \
--out_file "${OUTPUT_FILE}"
echo "Vertical data splits are generated in ${OUTPUT_PATH}"
