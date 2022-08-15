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
from custom.xgboost_learner import XGBoostLearner

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import make_reply
from nvflare.app_common.app_constant import AppConstants


class XGBoostHiggsLearner(XGBoostLearner):
    def __init__(
        self,
        train_config_filename,
        num_parallel_tree,
        lr_mode: str = "uniform",
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            train_task_name=train_task_name,
        )
        self.train_config_filename = train_config_filename
        self.num_parallel_tree = num_parallel_tree
        self.lr_mode = lr_mode

    def train_config(self, fl_ctx: FLContext):
        """HIGGS traning configuration
        Here, we use a json to specify the needed parameters
        """

        # Load training configurations json
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        data_path = self.config_info["data_path"]
        self.local_model_path = os.path.join(app_dir, self.config_info["local_model_path"])
        self.global_model_path = os.path.join(app_dir, self.config_info["global_model_path"])
        data_index = self.config_info["data_index"]
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
                f"Dict of data_index does not contain Validation split",
            )
            return make_reply(ReturnCode.TASK_ABORTED)
        site_index = data_index[self.client_id]
        valid_index = data_index["valid"]
        # use dynamic shrinkage - adjusted by personalized scaling factor
        if self.lr_mode not in ["uniform", "scaled"]:
            self.log_error(
                fl_ctx,
                f"Only support [uniform] or [scaled] mode, but got {self.lr_mode}",
            )
            return make_reply(ReturnCode.TASK_ABORTED)

        if self.num_parallel_tree > 1:
            # Bagging mode
            if self.num_parallel_tree != len(data_index.keys()) - 1:
                self.log_error(
                    fl_ctx,
                    f"Mismatch between num_parallel_tree and client number",
                )
                return make_reply(ReturnCode.TASK_ABORTED)
            if self.lr_mode == "uniform":
                # uniform lr, gloabl learnining_rate scaled by num_parallel_tree for bagging
                self.lr = self.config_info["learning_rate"] / self.num_parallel_tree
            else:
                # scaled lr, global learning_rate scaled by data size percentage
                self.lr = self.config_info["learning_rate"] * site_index["lr_scale"]
        else:
            # Cyclic mode
            if self.lr_mode == "uniform":
                # uniform lr, directly use the gloabl learning_rate
                self.lr = self.config_info["learning_rate"]
            else:
                # scaled lr, global learning_rate scaled by data size percentage
                # multiply by num_parallel_tree to recover the factor in lr_scale
                self.lr = self.config_info["learning_rate"] * site_index["lr_scale"] * self.num_parallel_tree

        self.objective = self.config_info["objective"]
        self.max_depth = self.config_info["max_depth"]
        self.eval_metric = self.config_info["eval_metric"]
        self.nthread = self.config_info["nthread"]

        # load data
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
