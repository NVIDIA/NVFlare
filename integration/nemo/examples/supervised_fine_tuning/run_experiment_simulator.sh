#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:${PWD}:/workspace/Libs/NeMo-1.18.0:/workspace/Libs/NVFlare-dev"
echo "PYTHONPATH is ${PYTHONPATH}"

pip install nvflare
pip install -r /workspace/Libs/NeMo-1.18.0/requirements/requirements.txt
pip install -r /workspace/Libs/NeMo-1.18.0/requirements/requirements_nlp.txt

nvflare simulator jobs/supervised_finetuning -w ${PWD}/workspace_sft -n 2 -gpu 0,1
