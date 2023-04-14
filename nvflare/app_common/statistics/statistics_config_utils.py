# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from nvflare.app_common.app_constant import StatisticsConstants as StC


def get_feature_bin_range(feature_name: str, hist_config: dict) -> Optional[List[float]]:
    bin_range = None
    if feature_name in hist_config:
        if StC.STATS_BIN_RANGE in hist_config[feature_name]:
            bin_range = hist_config[feature_name][StC.STATS_BIN_RANGE]
    elif "*" in hist_config:
        default_config = hist_config["*"]
        if StC.STATS_BIN_RANGE in default_config:
            bin_range = default_config[StC.STATS_BIN_RANGE]

    return bin_range
