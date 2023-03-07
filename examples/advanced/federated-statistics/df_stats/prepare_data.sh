#!/bin/bash

DATASET_PATH="/tmp/nvflare/df_stats/data"
python data_utils.py  --prepare-data --dest "${DATASET_PATH}"