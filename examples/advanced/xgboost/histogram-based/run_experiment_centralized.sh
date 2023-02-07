#!/usr/bin/env bash
DATASET_PATH="$HOME/dataset/HIGGS.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
    exit 1
fi
python3 ../utils/baseline_centralized.py --num_parallel_tree 1 --train_in_one_session --data_path "${DATASET_PATH}"
