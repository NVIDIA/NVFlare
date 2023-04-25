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

from typing import Optional, Tuple

from sklearn.metrics import roc_auc_score

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range
from nvflare.fuel.utils.import_utils import optional_import


class SVMLearner(Learner):
    def __init__(
        self,
        backend: str,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
    ):
        super().__init__()
        self.backend = backend
        if self.backend == "sklearn":
            self.svm_lib, flag = optional_import(module="sklearn.svm")
            if not flag:
                self.log_error(fl_ctx, "Can't import sklearn.svm")
                return
        elif self.backend == "cuml":
            self.svm_lib, flag = optional_import(module="cuml.svm")
            if not flag:
                self.log_error(fl_ctx, "Can't import cuml.svm")
                return
        else:
            self.system_panic(f"backend SVM library {self.backend} unknown!", fl_ctx)

        self.data_path = data_path
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end

        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.svm = None
        self.kernel = None
        self.params = {}

    def load_data(self) -> dict:
        train_data = load_data_for_range(self.data_path, self.train_start, self.train_end)
        valid_data = load_data_for_range(self.data_path, self.valid_start, self.valid_end)
        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]
        # model will be created after receiving global parameter of kernel

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        if curr_round == 0:
            # only perform training on the first round
            (x_train, y_train, train_size) = self.train_data
            self.kernel = global_param["kernel"]
            self.svm = self.svm_lib.SVC(kernel=self.kernel)
            # train model
            self.svm.fit(x_train, y_train)
            # get support vectors
            index = self.svm.support_
            local_support_x = x_train[index]
            local_support_y = y_train[index]
            self.params = {"support_x": local_support_x, "support_y": local_support_y}
        elif curr_round > 1:
            self.system_panic("Federated SVM only performs training for one round, system exiting.", fl_ctx)
        return self.params, self.svm

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # local validation with global support vectors
        # fit a standalone SVM with the global support vectors
        svm_global = self.svm_lib.SVC(kernel=self.kernel)
        support_x = global_param["support_x"]
        support_y = global_param["support_y"]
        svm_global.fit(support_x, support_y)
        # validate global model
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = svm_global.predict(x_valid)
        auc = roc_auc_score(y_valid, y_pred)
        self.log_info(fl_ctx, f"AUC {auc:.4f}")
        metrics = {"AUC": auc}
        return metrics, svm_global

    def finalize(self, fl_ctx: FLContext) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")
