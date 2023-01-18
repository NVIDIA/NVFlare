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

from typing import Optional

from app_opt.sklearn.sklearner import SKLearner
from nvflare.apis.fl_context import FLContext
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# Note: will move to this app_common when it gets matured


class SVMLearner(SKLearner):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
    ):
        super().__init__(data_path, train_start, train_end, valid_start, valid_end)
        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.svm = SVC(kernel="rbf")
        self.params = {}

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]

    def train(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        if curr_round == 0:
            # only perform training on the first round
            # the following rounds directly returns the retained records
            (x_train, y_train, train_size) = self.train_data
            self.svm.fit(x_train, y_train)
            index = self.svm.support_
            local_support_x = x_train[index]
            local_support_y = y_train[index]
            self.params = {"support_x": local_support_x, "support_y": local_support_y}
        return self.params, self.svm

    def evaluate(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        # local validation with global support vectors
        # fit a standalone SVM with the global support vectors
        svm_global = SVC(kernel="rbf")
        if global_param:
            support_x = global_param["support_x"]
            support_y = global_param["support_y"]
            svm_global.fit(support_x, support_y)
            (x_valid, y_valid, valid_size) = self.valid_data
            y_pred = svm_global.predict(x_valid)
            auc = roc_auc_score(y_valid, y_pred)
            metrics = {"AUC": auc}
        return metrics, svm_global

    def finalize(self) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(self.fl_ctx, "Freed training resources")
