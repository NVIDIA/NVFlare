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

import pandas as pd
from nvflare.app_common.app_constant import AppConstants
from sklearn_linear_executor import FedSKLearnLinearExecutor


def _read_CSV_with_pandas(data_path, start: int, end: int):
    data_size = end - start
    data = pd.read_csv(data_path, header=None, skiprows=start, nrows=data_size)
    data_num = data.shape[0]
    # split to feature and label
    X = data.iloc[:, 1:].copy()
    y = data.iloc[:, 0].copy()
    return X.to_numpy(), y.to_numpy(), data_num


class FedSKLearnLinearCSVExecutor(FedSKLearnLinearExecutor):
    def __init__(
        self,
        data_split_filename,
        local_model_path: str = "model.joblib",
        global_model_path: str = "model_global.joblib",
        learning_rate: str = "constant",
        eta0: float = 1e-4,
        loss: str = "log",
        penalty: str = "l2",
        fit_intercept: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__(
            local_model_path=local_model_path,
            global_model_path=global_model_path,
            learning_rate=learning_rate,
            eta0=eta0,
            loss=loss,
            penalty=penalty,
            fit_intercept=fit_intercept,
            train_task_name=train_task_name,
        )
        self.data_split_filename = data_split_filename

    def load_data(self):
        with open(self.data_split_filename) as file:
            data_split = json.load(file)
        data_path = data_split["data_path"]
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
        # training
        X_train, y_train, sample_size_train = _read_CSV_with_pandas(
            data_path=data_path, start=site_index["start"], end=site_index["end"]
        )
        # validation
        X_valid, y_valid, sample_size_valid = _read_CSV_with_pandas(
            data_path=data_path, start=valid_index["start"], end=valid_index["end"]
        )
        return X_train, y_train, X_valid, y_valid, sample_size_train
