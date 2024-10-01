#!/usr/bin/env bash
NB_DIR="/bionemo_nvflare_examples"
export PYTHONHASHSEED=0
pip install ujson
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --notebook-dir=${NB_DIR} --NotebookApp.allow_origin='*'
