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

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader


def _read_higgs_with_pandas(data_path, start: int, end: int):
    data_size = end - start
    data = pd.read_csv(data_path, header=None, skiprows=start, nrows=data_size)
    data_num = data.shape[0]

    # split to feature and label
    x = data.iloc[:, 1:].copy()
    y = data.iloc[:, 0].copy()

    return x, y, data_num


class HIGGSDataLoader(XGBDataLoader):
    def __init__(self, data_split_filename):
        """Reads HIGGS dataset and return XGB data matrix.

        Args:
            data_split_filename: file name to data splits
        """
        self.data_split_filename = data_split_filename

    def load_data(self, client_id: str):
        with open(self.data_split_filename, "r") as file:
            data_split = json.load(file)

        data_path = data_split["data_path"]
        data_index = data_split["data_index"]

        # check if site_id and "valid" in the mapping dict
        if client_id not in data_index.keys():
            raise ValueError(
                f"Data does not contain Client {client_id} split",
            )

        if "valid" not in data_index.keys():
            raise ValueError(
                "Data does not contain Validation split",
            )

        site_index = data_index[client_id]
        valid_index = data_index["valid"]

        # training
        x_train, y_train, total_train_data_num = _read_higgs_with_pandas(
            data_path=data_path, start=site_index["start"], end=site_index["end"]
        )
        dmat_train = xgb.DMatrix(x_train, label=y_train)

        # validation
        x_valid, y_valid, total_valid_data_num = _read_higgs_with_pandas(
            data_path=data_path, start=valid_index["start"], end=valid_index["end"]
        )
        dmat_valid = xgb.DMatrix(x_valid, label=y_valid)

        return dmat_train, dmat_valid
