#!/usr/bin/env bash
DATASET_ROOT=${1}
echo "BERT"
python3 ./utils/ner_model_test.py --model_path "/tmp/nvflare/workspaces/bert_ncbi/simulate_job/app_server/" --model_name "bert-base-uncased" --data_path ${DATASET_ROOT} --num_labels 3
echo "GPT-2"
python3 ./utils/ner_model_test.py --model_path "/tmp/nvflare/workspaces/gpt2_ncbi/simulate_job/app_server/" --model_name "gpt2" --data_path ${DATASET_ROOT} --num_labels 3
