#!/usr/bin/env bash
DATASET_PATH="$HOME/dataset/HIGGS.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

OUTPUT_PATH="data_splits"
if [ ! -d ${OUTPUT_PATH} ]
then
    mkdir ${OUTPUT_PATH}
fi

for site_num in 2 5 20;
do
    for split_mode in uniform exponential square;
    do
        python3 utils/prepare_data_split.py \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/data_split_${site_num}_${split_mode}.json"
    done
done
