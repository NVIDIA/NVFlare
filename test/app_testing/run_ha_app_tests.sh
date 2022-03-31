#!/usr/bin/env bash

pip install tensorflow

python test_runner.py \
    --poc ../../nvflare/poc \
    --n_clients 2 \
    --yaml test_ha_apps.yml \
    --app_path test_apps \
    --snapshot_path /tmp/snapshot-storage \
    --ha \
    --cleanup
