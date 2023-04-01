#!/usr/bin/env bash
DATASET_ROOT=${1}
python3 ./utils/bert_ner_model_test.py --model_path "/tmp/nvflare/workspaces/bert_ncbi/simulate_job/app_server/" --data_path ${DATASET_ROOT} --num_labels 3
