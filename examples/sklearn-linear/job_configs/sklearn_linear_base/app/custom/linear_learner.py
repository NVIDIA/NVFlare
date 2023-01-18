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

import copy
import warnings
from typing import Optional

import numpy as np
from app_opt.sklearn.sklearner import SKLearner
from nvflare.apis.fl_context import FLContext
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

# Note: will move to this app_common when it gets matured


class LinearLearner(SKLearner):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
        learning_rate: str = "constant",
        eta0: float = 1e-4,
        loss: str = "log",
        penalty: str = "l2",
        fit_intercept: int = 1,
    ):
        super().__init__(data_path, train_start, train_end, valid_start, valid_end)
        self.train_data = None
        self.valid_data = None
        self.n_samples = None

        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.loss = loss
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]

        # initialize model to all zero
        self.local_model = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            fit_intercept=bool(self.fit_intercept),
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            max_iter=1,
            warm_start=True,
            random_state=0,
        )
        n_classes = 2  # Binary classification
        n_features = data["train"][0].shape[1]  # Number of features in dataset
        self.local_model.classes_ = np.array([i for i in range(n_classes)])
        self.local_model.coef_ = np.zeros((1, n_features))
        if self.fit_intercept:
            self.local_model.intercept_ = np.zeros((1,))

    def set_parameters(self, params):
        self.local_model.coef_ = params["coef"]
        if self.local_model.fit_intercept:
            self.local_model.intercept_ = params["intercept"]

    def train(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        # get training data
        (x_train, y_train, train_size) = self.train_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.local_model.fit(x_train, y_train)
        if self.local_model.fit_intercept:
            params = {
                "coef": self.local_model.coef_,
                "intercept": self.local_model.intercept_,
            }
        else:
            params = {"coef": self.local_model.coef_}
        return copy.deepcopy(params), self.local_model

    def evaluate(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        # set local model with global parameters
        # and perform validation
        if global_param:
            self.set_parameters(global_param)
            (x_valid, y_valid, valid_size) = self.valid_data
            y_pred = self.local_model.predict(x_valid)
            auc = roc_auc_score(y_valid, y_pred)
            metrics = {"AUC": auc}
        return metrics, self.local_model

    def finalize(self) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(self.fl_ctx, "Freed training resources")
