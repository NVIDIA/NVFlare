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

import json

import pandas as pd
import xgboost as xgb

from nvflare.app_opt.xgboost.histogram_based.executor import FedXGBHistogramExecutor


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

    def load_data(self):
        with open(self.data_split_filename, "r") as file:
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
        X_train, y_train, total_train_data_num = _read_HIGGS_with_pandas(
            data_path=data_path, start=site_index["start"], end=site_index["end"]
        )
        dmat_train = xgb.DMatrix(X_train, label=y_train)

        # validation
        X_valid, y_valid, total_valid_data_num = _read_HIGGS_with_pandas(
            data_path=data_path, start=valid_index["start"], end=valid_index["end"]
        )
        dmat_valid = xgb.DMatrix(X_valid, label=y_valid)

        return dmat_train, dmat_valid
