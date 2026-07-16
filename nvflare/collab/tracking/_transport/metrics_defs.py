# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Constants for Collab tracking/metrics system."""

# CellNet channel and topic for metrics
METRICS_CHANNEL = "collab_metrics"
METRICS_TOPIC = "log"


# Metrics message keys
class MetricsKey:
    KEY = "key"
    VALUE = "value"
    DATA_TYPE = "data_type"
    STEP = "step"
    EPOCH = "epoch"
    GLOBAL_STEP = "global_step"
    SITE_NAME = "site_name"
    RANK = "rank"


# Environment variable for metrics endpoint
ENV_METRICS_ENABLED = "COLLAB_METRICS_ENABLED"
