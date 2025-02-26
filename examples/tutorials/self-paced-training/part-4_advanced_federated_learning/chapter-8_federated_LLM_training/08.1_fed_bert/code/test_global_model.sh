#!/usr/bin/env bash
DATASET_ROOT=${1}
echo "BERT"
python3 ner_model_test.py --model_path "/tmp/nvflare/workspace/works/Bert/server/simulate_job/app_server/" --model_name "bert-base-uncased" --data_path ${DATASET_ROOT} --num_labels 3