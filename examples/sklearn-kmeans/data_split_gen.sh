#!/usr/bin/env bash
DATASET_PATH="${PWD}/dataset/sklearn_iris.csv"
OUTPUT_PATH="/tmp/nvflare/iris_dataset"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved Iris dataset in ${DATASET_PATH}"
fi

data_size=150
validation_size=150
echo "Generated Iris data splits, reading from ${DATASET_PATH}"

for site_num in 3;
do
    for split_mode in uniform linear;
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
