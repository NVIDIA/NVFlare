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

import xgboost as xgb

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based.executor import FedXGBHistogramExecutor, TensorBoardCallback, XGBoostParams
from nvflare.fuel.utils.import_utils import optional_import


class XGBoostTrainer(FedXGBHistogramExecutor):
    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
        self.data_loader = engine.get_component(self.data_loader_id)
        if not isinstance(self.data_loader, XGBDataLoader):
            self.system_panic("data_loader should be type XGBDataLoader", fl_ctx)
        self.fl_ctx = fl_ctx

    def xgb_train(self, params: XGBoostParams) -> xgb.core.Booster:
        # Load the DMatrices from the DataLoader
        dtrain, dvalid = self.data_loader.load_data(self.fl_ctx)

        # Specify validations set to watch performance
        watchlist = [(dvalid, "eval"), (dtrain, "train")]

        # Add optional tensorboard callback
        callbacks = [xgb.callback.EvaluationMonitor(rank=self.rank)]
        tensorboard, flag = optional_import(module="torch.utils.tensorboard")
        if flag and self.app_dir:
            callbacks.append(TensorBoardCallback(self.app_dir, tensorboard))

        # Run training, all the features in training API is available.
        bst = xgb.train(
            params.xgb_params,
            dtrain,
            params.num_rounds,
            evals=watchlist,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose_eval=params.verbose_eval,
            callbacks=callbacks,
        )

        return bst
