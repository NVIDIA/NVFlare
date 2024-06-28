#!/usr/bin/env bash
directory="/tmp/nvflare/xgb_exp"

if [ ! -e "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi

echo "Training baseline"
python3 ./train_central/train_base.py 3
