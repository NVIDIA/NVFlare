#!/usr/bin/env bash
DATASET_PATH="/tmp/nvflare/dataset/HIGGS.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

valid_frac=0.1
echo "Generating job configs with data splits, reading from ${DATASET_PATH}"

task_name="sklearn_linear"
for site_num in 5;
do
    for split_mode in uniform;
    do
        python3 utils/prepare_job_config.py \
        --task_name "${task_name}" \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --valid_frac ${valid_frac} \
        --split_method ${split_mode}
    done
done
