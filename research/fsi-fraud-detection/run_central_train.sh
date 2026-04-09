#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


BASE_RESULTS_DIR="results"

epochs=20
bs=64
lr=5e-4
wf=1
dp=0.1
gamma=2.0
results_dir="${BASE_RESULTS_DIR}/central_train_${lr}_epochs${epochs}_bs${bs}_wf${wf}_dp${dp}_gamma${gamma}"

echo "[$(date +%H:%M:%S)] Starting: lr=$lr epochs=$epochs bs=$bs wf=$wf dp=$dp gamma=$gamma"
uv run python -m train.central_train \
    --data_selection central-exp \
    --epochs "$epochs" \
    --batch-size "$bs" \
    --lr "$lr" \
    --width_factor="$wf" \
    --dropout_p="$dp" \
    --focal_gamma="$gamma" \
    --results_dir "$results_dir"

