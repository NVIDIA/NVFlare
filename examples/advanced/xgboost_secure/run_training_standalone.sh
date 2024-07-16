#!/usr/bin/env bash

directory="/tmp/nvflare/xgb_exp"
if [ ! -e "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi

echo "Training baseline"
python3 ./train_standalone/train_base.py
echo "Training horizontal"
python3 ./train_standalone/train_hori_base.py 3
echo "Training secure horizontal"
python3 ./train_standalone/train_hori_secure.py 3
echo "Training vertical"
python3 ./train_standalone/train_vert_base.py 3
echo "Training secure vertical"
python3 ./train_standalone/train_vert_secure.py 3

echo "Training baseline GPU"
python3 ./train_standalone/train_base_gpu.py
echo "Training horizontal GPU"
python3 ./train_standalone/train_hori_base_gpu.py 3