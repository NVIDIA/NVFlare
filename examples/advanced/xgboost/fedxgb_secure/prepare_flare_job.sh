#!/usr/bin/env bash


# config the job template directory
nvflare config -jt ../../../job_templates/

# create horizontal job
nvflare job create -force -w xgboost -j ./jobs/xgb_hori \
    -f config_fed_server.conf secure_training=false data_split_mode=0 \
    -f config_fed_client.conf folder="/tmp/nvflare/xgb_dataset/horizontal_xgb_data"

# create horizontal secure job
nvflare job create -force -w xgboost -j ./jobs/xgb_hori_secure \
    -f config_fed_server.conf secure_training=true data_split_mode=0 \
    -f config_fed_client.conf folder="/tmp/nvflare/xgb_dataset/horizontal_xgb_data"

# create vertical job
nvflare job create -force -w xgboost -j ./jobs/xgb_vert \
    -f config_fed_server.conf secure_training=false data_split_mode=1 \
    -f config_fed_client.conf folder="/tmp/nvflare/xgb_dataset/vertical_xgb_data"

# create vertical secure job
nvflare job create -force -w xgboost -j ./jobs/xgb_vert_secure \
    -f config_fed_server.conf secure_training=true data_split_mode=1 \
    -f config_fed_client.conf folder="/tmp/nvflare/xgb_dataset/vertical_xgb_data"
