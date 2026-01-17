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

import warnings

warnings.warn(
    "The 'nvflare.app_opt.xgboost.histogram_based' module (V1) is deprecated and will be removed in "
    "version 2.8.0. Please use 'nvflare.app_opt.xgboost.histogram_based_v2' instead, or set "
    "algorithm='histogram_v2' in XGBHistogramRecipe.",
    DeprecationWarning,
    stacklevel=2,
)
