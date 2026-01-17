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


class CSVDataLoader(XGBDataLoader):
    def __init__(self, folder: str):
        """Reads CSV dataset and returns XGB data matrix with automatic client-specific loading.

        This data loader automatically handles site-specific data loading. Even though you pass
        the same folder path to all clients, each client will load its own data based on its
        client_id (which is injected by the framework at runtime).

        Expected folder structure:
            {folder}/
            ├── site-1/
            │   ├── train.csv
            │   └── valid.csv
            ├── site-2/
            │   ├── train.csv
            │   └── valid.csv
            └── site-3/
                ├── train.csv
                └── valid.csv

        For horizontal mode (row split):
            - Each site's CSV contains all features + labels
            - Each site has different rows (samples)

        For vertical mode (column split):
            - site-1 (rank 0) contains subset of features + labels
            - Other sites contain different features, no labels
            - All sites have the same rows (samples)

        Args:
            folder: Base folder path containing client-specific subdirectories.
                Each client will automatically load from {folder}/{client_id}/

        Example:
            .. code-block:: python

                # In your job script - same data loader for all clients
                for i in range(1, 4):
                    dataloader = CSVDataLoader(folder="/tmp/data/horizontal")
                    recipe.add_to_client(f"site-{i}", dataloader)

                # At runtime:
                # site-1 loads: /tmp/data/horizontal/site-1/train.csv
                # site-2 loads: /tmp/data/horizontal/site-2/train.csv
                # site-3 loads: /tmp/data/horizontal/site-3/train.csv

        Note:
            In vertical mode, the label owner is always rank 0 (typically site-1).
        """
        self.folder = folder

    def load_data(self):

        train_path = f"{self.folder}/{self.client_id}/train.csv"
        valid_path = f"{self.folder}/{self.client_id}/valid.csv"

        if self.rank == 0 or self.data_split_mode == xgb.core.DataSplitMode.ROW:
            label = "&label_column=0"
        else:
            label = ""

        train_data = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)
        valid_data = xgb.DMatrix(valid_path + f"?format=csv{label}", data_split_mode=self.data_split_mode)

        return train_data, valid_data
