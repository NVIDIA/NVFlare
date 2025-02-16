#!/usr/bin/env bash

directory="/tmp/nvflare/xgb_exp"
if [ ! -e "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi

echo "Training baseline CPU"
python3 ./train_standalone/train_base.py --out_path "/tmp/nvflare/xgb_exp/base_cpu" --gpu 0
echo "Training baseline GPU"
python3 ./train_standalone/train_base.py --out_path "/tmp/nvflare/xgb_exp/base_gpu" --gpu 1
echo "Training horizontal CPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/horizontal_xgb_data" --out_path "/tmp/nvflare/xgb_exp/hori_cpu_non_enc" --vert 0 --gpu 0 --enc 0
echo "Training horizontal CPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/horizontal_xgb_data" --out_path "/tmp/nvflare/xgb_exp/hori_cpu_enc" --vert 0 --gpu 0 --enc 1
echo "Training horizontal GPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/horizontal_xgb_data" --out_path "/tmp/nvflare/xgb_exp/hori_gpu_non_enc" --vert 0 --gpu 1 --enc 0
echo "Training horizontal GPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/horizontal_xgb_data" --out_path "/tmp/nvflare/xgb_exp/hori_gpu_enc" --vert 0 --gpu 1 --enc 1
echo "Training vertical CPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/vertical_xgb_data" --out_path "/tmp/nvflare/xgb_exp/vert_cpu_non_enc" --vert 1 --gpu 0 --enc 0
echo "Training vertical CPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/vertical_xgb_data" --out_path "/tmp/nvflare/xgb_exp/vert_cpu_enc" --vert 1 --gpu 0 --enc 1
echo "Training vertical GPU non-encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/vertical_xgb_data" --out_path "/tmp/nvflare/xgb_exp/vert_gpu_non_enc" --vert 1 --gpu 1 --enc 0
echo "Training vertical GPU encrypted"
python3 ./train_standalone/train_federated.py --data_train_root "/tmp/nvflare/xgb_dataset/vertical_xgb_data" --out_path "/tmp/nvflare/xgb_exp/vert_gpu_enc" --vert 1 --gpu 1 --enc 1