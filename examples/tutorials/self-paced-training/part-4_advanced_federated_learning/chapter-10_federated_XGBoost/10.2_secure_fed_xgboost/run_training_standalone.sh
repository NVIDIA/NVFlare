#!/usr/bin/env bash

workspace_root="/tmp/nvflare/workspace/fedxgb_secure/train_standalone/"
if [ ! -e "$workspace_root" ]; then
    mkdir -p "$workspace_root"
    echo "Directory created: $workspace_root"
else
    echo "Directory already exists: $workspace_root"
fi

dataset_path="/tmp/nvflare/dataset/xgb_dataset/"

echo "Training baseline CPU"
python3 ./train_standalone/train_base.py --out_path "$workspace_root/base_cpu" --gpu 0
echo "Training baseline GPU"
python3 ./train_standalone/train_base.py --out_path "$workspace_root/base_gpu" --gpu 1
echo "Training horizontal CPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/horizontal_xgb_data" --out_path "$workspace_root/hori_cpu_non_enc" --vert 0 --gpu 0 --enc 0
echo "Training horizontal CPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/horizontal_xgb_data" --out_path "$workspace_root/hori_cpu_enc" --vert 0 --gpu 0 --enc 1
echo "Training horizontal GPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/horizontal_xgb_data" --out_path "$workspace_root/hori_gpu_non_enc" --vert 0 --gpu 1 --enc 0
echo "Training horizontal GPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/horizontal_xgb_data" --out_path "$workspace_root/hori_gpu_enc" --vert 0 --gpu 1 --enc 1
echo "Training vertical CPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/vertical_xgb_data" --out_path "$workspace_root/vert_cpu_non_enc" --vert 1 --gpu 0 --enc 0
echo "Training vertical CPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/vertical_xgb_data" --out_path "$workspace_root/vert_cpu_enc" --vert 1 --gpu 0 --enc 1
echo "Training vertical GPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/vertical_xgb_data" --out_path "$workspace_root/vert_gpu_non_enc" --vert 1 --gpu 1 --enc 0
echo "Training vertical GPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "$dataset_path/vertical_xgb_data" --out_path "$workspace_root/vert_gpu_enc" --vert 1 --gpu 1 --enc 1