#!/usr/bin/env bash
OUTPUT_PATH="/tmp/nvflare/breast_cancer_dataset"
data_size=569
validation_size=116
echo "Generated Breast Cancer data splits, with total size ${data_size} and validation size ${}"

for site_num in 3;
do
    for split_mode in uniform linear;
    do
        python3 utils/prepare_data_split.py \
        --size_total ${data_size} \
        --size_valid ${validation_size} \
        --site_num ${site_num} \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Data splits are generated in ${OUTPUT_PATH}"
