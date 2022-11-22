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
from sklearn_kmeans_executor import FedSKLearnKMeansExecutor

from nvflare.app_common.app_constant import AppConstants


def _load_Iris_with_permutation():
    # Load data directly
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Randomize with fixed seed so that there won't be overlapping across clients
    total_sample_size = X.shape[0]
    np.random.seed(0)
    idx_random = np.random.permutation(total_sample_size)
    X_random = X[idx_random, :]
    y_random = y[idx_random]
    return X_random, y_random


class FedSKLearnKMeansIrisExecutor(FedSKLearnKMeansExecutor):
    def __init__(
        self,
        data_split_filename,
        local_model_path: str = "model.joblib",
        global_model_path: str = "model_global.joblib",
        n_clusters: int = 2,
        subsample_rate: float = 0.2,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            local_model_path=local_model_path,
            global_model_path=global_model_path,
            n_clusters=n_clusters,
            train_task_name=train_task_name,
        )
        self.subsample_rate = subsample_rate
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
        X, y = _load_Iris_with_permutation()
        # Get local training set
        # With subsampling
        X_train = X[site_index["start"] : site_index["end"], :]
        y_train = y[site_index["start"] : site_index["end"]]
        if self.subsample_rate < 1:
            idx = np.random.choice(np.arange(len(y_train)), int(len(y_train) * self.subsample_rate), replace=False)
            X_train = X_train[idx, :]
            y_train = y_train[idx]
        # Get local validation set
        X_valid = X[valid_index["start"] : valid_index["end"], :]
        y_valid = y[valid_index["start"] : valid_index["end"]]

        return X_train, y_train, X_valid, y_valid
