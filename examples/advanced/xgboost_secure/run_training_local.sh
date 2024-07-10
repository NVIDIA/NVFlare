#!/usr/bin/env bash
mkdir /tmp/nvflare/xgb_exp

echo "Training baseline"
python3 ./train_local/train_base.py
echo "Training horizontal"
python3 ./train_local/train_hori_base.py 3
echo "Training secure horizontal"
python3 ./train_local/train_hori_secure.py 3
echo "Training vertical"
python3 ./train_local/train_vert_base.py 3
echo "Training secure vertical"
python3 ./train_local/train_vert_secure.py 3