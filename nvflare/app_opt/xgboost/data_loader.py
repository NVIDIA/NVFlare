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
