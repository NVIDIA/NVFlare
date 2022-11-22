#!/usr/bin/env bash
OUTPUT_PATH="/tmp/nvflare/iris_dataset"
data_size=150

echo "Generated Iris data splits, with size ${data_size}"

for site_num in 3;
do
    for split_mode in uniform linear;
    do
        python3 utils/prepare_data_split.py \
        --data_size ${data_size} \
        --site_num ${site_num} \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Data splits are generated in ${OUTPUT_PATH}"
