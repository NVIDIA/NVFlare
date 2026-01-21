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


from abc import ABC, abstractmethod
from typing import Tuple

import xgboost as xgb


class XGBDataLoader(ABC):
    def initialize(
        self, client_id: str, rank: int, data_split_mode: xgb.core.DataSplitMode = xgb.core.DataSplitMode.ROW
    ):
        """Initialize the data loader with client-specific parameters.

        This method is automatically called by the NVFlare framework at runtime for each client.
        You do not need to call this method manually. The framework will inject the appropriate
        client_id, rank, and data_split_mode for each client site.

        Args:
            client_id: Unique identifier for the client (e.g., 'site-1', 'site-2').
                This is set automatically by the framework based on which client is running.
            rank: Client's rank in the federation (0-indexed). In vertical mode, rank 0
                is typically the label owner.
            data_split_mode: XGBoost data split mode (ROW for horizontal, COLUMN for vertical).
                This is set automatically based on the recipe configuration.

        Note:
            Even though you create the same data loader instance for all clients in your job script,
            each client will receive different values for these parameters at runtime, enabling
            client-specific data loading.
        """
        self._client_id = client_id
        self._rank = rank
        self._data_split_mode = data_split_mode

    @property
    def client_id(self):
        return self._client_id

    @property
    def rank(self):
        return self._rank

    @property
    def data_split_mode(self):
        return self._data_split_mode

    @abstractmethod
    def load_data(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """Loads data for xgboost.

        Returns:
            A tuple of train_data, validation_data
        """
        pass
