# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class XGBoostParams:
    def __init__(self, xgb_params: dict, num_rounds=10, early_stopping_rounds=2, verbose_eval=False):
        """Container for all XGBoost parameters.

        Args:
            xgb_params: This dict is passed to `xgboost.train()` as the first argument `params`.
                It contains all the Booster parameters.
                Please refer to XGBoost documentation for details:
                https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
        """
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.xgb_params: dict = xgb_params if xgb_params else {}
