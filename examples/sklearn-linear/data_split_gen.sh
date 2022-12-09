#!/usr/bin/env bash
DATASET_PATH="/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv"
OUTPUT_PATH="/tmp/nvflare/higgs_dataset"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

data_size=11000000
validation_size=1000000
echo "Generated HIGGS data splits, reading from ${DATASET_PATH}"

for site_num in 5 20;
do
    for split_mode in uniform exponential square;
    do
        python3 utils/prepare_data_split.py \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --size_total ${data_size} \
        --size_valid ${validation_size} \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Data splits are generated in ${OUTPUT_PATH}"
