# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Tuple

import xgboost as xgb
from xgboost import callback

from nvflare.apis.fl_context import FLContext

from .executor_spec import FedXGBHistogramExecutorSpec


class XGBoostParams:
    """Container for all XGBoost parameters"""

    def __init__(self, xgb_params: dict, num_rounds=10, early_stopping_rounds=2, verbose_eval=False):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.xgb_params: dict = xgb_params if xgb_params else {}


class FedXGBHistogramExecutor(FedXGBHistogramExecutorSpec, ABC):
    """Federated XGBoost Executor Spec for histogram-base collaboration.

    This class implements the basic xgb_train logic, the subclass must implement load_data.
    """

    @abstractmethod
    def load_data(self, fl_ctx: FLContext) -> Tuple[xgb.core.DMatrix, xgb.core.DMatrix]:
        raise NotImplementedError

    def xgb_train(self, params: XGBoostParams) -> xgb.core.Booster:
        # Load file, file will not be sharded in federated mode.
        dtrain = self.dmat_train
        dval = self.dmat_valid

        # Specify validations set to watch performance
        watchlist = [(dval, "eval"), (dtrain, "train")]

        # Run training, all the features in training API is available.
        bst = xgb.train(
            params.xgb_params,
            dtrain,
            params.num_rounds,
            evals=watchlist,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose_eval=params.verbose_eval,
            callbacks=[callback.EvaluationMonitor(rank=self.rank)],
        )

        return bst
