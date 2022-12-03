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

import numpy as np
from sklearn import datasets
from sklearn_svm_executor import FedSKLearnSVMExecutor

from nvflare.app_common.app_constant import AppConstants


def _load_Breast_Cancer():
    # Load data directly
    iris = datasets.load_breast_cancer()
    X = iris.data
    y = iris.target
    return X, y


class FedSKLearnSVMBreastCencerExecutor(FedSKLearnSVMExecutor):
    def __init__(
        self,
        data_split_filename,
        local_model_path: str = "model.joblib",
        global_model_path: str = "model_global.joblib",
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            local_model_path=local_model_path,
            global_model_path=global_model_path,
            train_task_name=train_task_name,
        )
        self.data_split_filename = data_split_filename

    def load_data(self):
        with open(self.data_split_filename) as file:
            data_split = json.load(file)
        data_index = data_split["data_index"]
        # check if site_id and "valid" in the mapping dict
        if self.client_id not in data_index.keys():
            raise ValueError(
                f"Dict of data_index does not contain Client {self.client_id} split",
            )
        if "valid" not in data_index.keys():
            raise ValueError(
                "Dict of data_index does not contain Validation split",
            )
        site_index = data_index[self.client_id]
        valid_index = data_index["valid"]
        # Load data
        X, y = _load_Breast_Cancer()
        # Get local training set
        # With subsampling
        X_train = X[site_index["start"] : site_index["end"], :]
        y_train = y[site_index["start"] : site_index["end"]]
        # Get local validation set
        X_valid = X[valid_index["start"] : valid_index["end"], :]
        y_valid = y[valid_index["start"] : valid_index["end"]]

        return X_train, y_train, X_valid, y_valid
