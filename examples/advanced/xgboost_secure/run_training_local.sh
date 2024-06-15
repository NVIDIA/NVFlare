#!/usr/bin/env bash
mkdir /tmp/nvflare/xgb_exp

echo "Training baseline"
python3 ./train_local/train_base.py 3