# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import warnings

warnings.warn(
    f"This module: {__file__} is deprecated. Please use nvflare.app_opt.he.intime_accumulate_model_aggregator",
    category=FutureWarning,
    stacklevel=2,
)

# flake8: noqa: F401
from nvflare.app_opt.he.intime_accumulate_model_aggregator import HEInTimeAccumulateWeightedAggregator
