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

import json
import os

import pandas as pd
import xgboost as xgb
from xgboost import callback

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_opt.xgboost.histogram_based.executor import FedXGBHistogramExecutor, XGBoostParams


def _read_HIGGS_with_pandas(data_path, start: int, end: int):
    data_size = end - start
    data = pd.read_csv(data_path, header=None, skiprows=start, nrows=data_size)
    data_num = data.shape[0]

    # split to feature and label
    x = data.iloc[:, 1:].copy()
    y = data.iloc[:, 0].copy()

    return x, y, data_num


class FedXGBHistogramHiggsExecutor(FedXGBHistogramExecutor):
    def __init__(self, data_split_filename, num_rounds, early_stopping_round, xgboost_params, verbose_eval=False):
        """Federated XGBoost Executor for histogram-base collaboration.

        This class sets up the training environment for Federated XGBoost. This is an abstract class and xgb_train
        method must be implemented by a subclass.

        Args:
            data_split_filename: file name to data splits
            num_rounds: number of boosting rounds
            early_stopping_round: early stopping round
            xgboost_params: parameters to passed in xgb
            verbose_eval: verbose_eval in xgb
        """
        super().__init__(num_rounds, early_stopping_round, xgboost_params, verbose_eval)
        self.data_split_filename = data_split_filename
        self.train_data = None
        self.test_data = None

    def load_data(self, fl_ctx: FLContext):
        """Loads data."""
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        client_id = fl_ctx.get_identity_name()

        data_split_file_path = os.path.join(app_config_dir, self.data_split_filename)
        with open(data_split_file_path) as file:
            data_split = json.load(file)

        data_path = data_split["data_path"]
        data_index = data_split["data_index"]

        # check if site_id and "valid" in the mapping dict
        if client_id not in data_index.keys():
            raise ValueError(
                f"Dict of data_index does not contain Client {client_id} split",
            )

        if "valid" not in data_index.keys():
            raise ValueError(
                "Dict of data_index does not contain Validation split",
            )

        site_index = data_index[client_id]
        valid_index = data_index["valid"]

        # training
        X_train, y_train, total_train_data_num = _read_HIGGS_with_pandas(
            data_path=data_path, start=site_index["start"], end=site_index["end"]
        )
        dmat_train = xgb.DMatrix(X_train, label=y_train)

        # validation
        X_valid, y_valid, total_valid_data_num = _read_HIGGS_with_pandas(
            data_path=data_path, start=valid_index["start"], end=valid_index["end"]
        )
        dmat_valid = xgb.DMatrix(X_valid, label=y_valid)

        self.log_info(
            fl_ctx,
            f"Total training/validation data count: {total_train_data_num}/{total_valid_data_num}",
        )

        self.train_data = dmat_train
        self.test_data = dmat_valid

    def xgb_train(self, params: XGBoostParams, fl_ctx: FLContext) -> Shareable:
        # Load file, file will not be sharded in federated mode.
        dtrain = self.train_data
        dtest = self.test_data

        # Specify validations set to watch performance
        watchlist = [(dtest, "eval"), (dtrain, "train")]

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

        # Save the model.
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = workspace.get_run_dir(run_number)
        bst.save_model(os.path.join(run_dir, "test.model.json"))

        return make_reply(ReturnCode.OK)
