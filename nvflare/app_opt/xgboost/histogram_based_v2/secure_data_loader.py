# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import xgboost as xgb

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader

COL_SECURE = 2
ROW_SECURE = 3


class SecureDataLoader(XGBDataLoader):
    def __init__(self, rank: int, folder: str, data_split_mode=COL_SECURE):
        """Reads CSV dataset and return XGB data matrix in vertical secure mode.

        Args:
            rank: Rank of the site
            folder: Folder to find the CSV files
        """
        self.rank = rank
        self.folder = folder
        self.data_split_mode = data_split_mode

    def load_data(self, client_id: str):

        train_path = f"{self.folder}/site-{self.rank + 1}/train.csv"
        valid_path = f"{self.folder}/site-{self.rank + 1}/valid.csv"

        if self.rank == 0 or self.data_split_mode == ROW_SECURE:
            label = "&label_column=0"
        else:
            label = ""

        train_data = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)
        valid_data = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)

        return train_data, valid_data
