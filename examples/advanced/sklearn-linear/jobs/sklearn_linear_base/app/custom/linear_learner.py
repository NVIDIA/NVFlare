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

import copy
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range


class LinearLearner(Learner):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
        random_state: int = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end

        self.random_state = random_state
        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.local_model = None
        self.n_features = None

    def load_data(self) -> dict:
        train_data = load_data_for_range(self.data_path, self.train_start, self.train_end)
        valid_data = load_data_for_range(self.data_path, self.valid_start, self.valid_end)
        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Loading data from {self.data_path}")
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential aggregation
        self.n_samples = data["train"][-1]
        self.n_features = data["train"][0].shape[1]
        # model will be created after receiving global parameters

    def set_parameters(self, params):
        self.local_model.coef_ = params["coef"]
        if self.local_model.fit_intercept:
            self.local_model.intercept_ = params["intercept"]

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        (x_train, y_train, train_size) = self.train_data
        if curr_round == 0:
            # initialize model with global_param
            # and set to all zero
            fit_intercept = bool(global_param["fit_intercept"])
            self.local_model = SGDClassifier(
                loss=global_param["loss"],
                penalty=global_param["penalty"],
                fit_intercept=fit_intercept,
                learning_rate=global_param["learning_rate"],
                eta0=global_param["eta0"],
                max_iter=1,
                warm_start=True,
                random_state=self.random_state,
            )
            n_classes = global_param["n_classes"]
            self.local_model.classes_ = np.array(list(range(n_classes)))
            self.local_model.coef_ = np.zeros((1, self.n_features))
            if fit_intercept:
                self.local_model.intercept_ = np.zeros((1,))
        # Training starting from global model
        # Note that the parameter update using global model has been performed
        # during global model evaluation
        self.local_model.fit(x_train, y_train)
        if self.local_model.fit_intercept:
            params = {
                "coef": self.local_model.coef_,
                "intercept": self.local_model.intercept_,
            }
        else:
            params = {"coef": self.local_model.coef_}
        return copy.deepcopy(params), self.local_model

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # set local model with global parameters
        self.set_parameters(global_param)
        # perform validation
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = self.local_model.predict(x_valid)
        auc = roc_auc_score(y_valid, y_pred)
        self.log_info(fl_ctx, f"AUC {auc:.4f}")
        metrics = {"AUC": auc}
        return metrics, self.local_model

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")
