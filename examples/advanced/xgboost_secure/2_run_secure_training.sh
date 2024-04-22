#!/bin/bash
cp /media/ziyuexu/Research/Experiment/SecureFedXGBoost/FL_Exp/NVFlare-secure_xgb_integration/integration/xgboost/processor/build/libproc_nvflare.so /tmp

set -e
mkdir -p ./model
world_size=2
python train_federated_ver_secure.py "${world_size}"
