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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.learners.xgboost_tree_fed_learner import XGBoostTreeFedLearner


class XGBoostTreeFedHiggsLearner(XGBoostTreeFedLearner):
    def __init__(
        self,
        data_split_filename,
        training_mode,
        num_tree_bagging: int = 1,
        lr_mode: str = "uniform",
        local_model_path: str = "model.json",
        global_model_path: str = "model_global.json",
        learning_rate: float = 0.1,
        objective: str = "binary:logistic",
        max_depth: int = 8,
        eval_metric: str = "auc",
        nthread: int = 16,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            training_mode=training_mode,
            num_tree_bagging=num_tree_bagging,
            lr_mode=lr_mode,
            local_model_path=local_model_path,
            global_model_path=global_model_path,
            learning_rate=learning_rate,
            objective=objective,
            max_depth=max_depth,
            eval_metric=eval_metric,
            nthread=nthread,
            train_task_name=train_task_name,
        )
        self.data_split_filename = data_split_filename

    def load_data(self, fl_ctx: FLContext):
        # load data split file for HIGGS
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())

        data_split_file_path = os.path.join(app_config_dir, self.data_split_filename)
        with open(data_split_file_path) as file:
            self.data_split = json.load(file)

        data_path = self.data_split["data_path"]
        data_index = self.data_split["data_index"]
        # check if site_id and "valid" in the mapping dict
        if self.client_id not in data_index.keys():
            self.log_error(
                fl_ctx,
                f"Dict of data_index does not contain Client {self.client_id} split",
            )
            return make_reply(ReturnCode.TASK_ABORTED)
        if "valid" not in data_index.keys():
            self.log_error(
                fl_ctx,
                "Dict of data_index does not contain Validation split",
            )
            return make_reply(ReturnCode.TASK_ABORTED)
        site_index = data_index[self.client_id]
        valid_index = data_index["valid"]
        # training
        start = site_index["start"]
        end = site_index["end"]
        data_size = end - start
        higgs = pd.read_csv(data_path, header=None, skiprows=start, nrows=data_size)
        total_train_data_num = higgs.shape[0]
        # split to feature and label
        X_higgs_train = higgs.iloc[:, 1:]
        y_higgs_train = higgs.iloc[:, 0]
        # validation
        start = valid_index["start"]
        end = valid_index["end"]
        higgs = pd.read_csv(data_path, header=None, skiprows=start, nrows=end - start)
        total_valid_data_num = higgs.shape[0]
        self.log_info(
            fl_ctx,
            f"Total training/validation data count: {total_train_data_num}/{total_valid_data_num}",
        )
        # split to feature and label
        X_higgs_valid = higgs.iloc[:, 1:]
        y_higgs_valid = higgs.iloc[:, 0]

        # construct xgboost DMatrix
        self.dmat_train = xgb.DMatrix(X_higgs_train, label=y_higgs_train)
        self.dmat_valid = xgb.DMatrix(X_higgs_valid, label=y_higgs_valid)
        self.valid_y = y_higgs_valid
