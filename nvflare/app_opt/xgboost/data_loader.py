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
    @abstractmethod
    def load_data(self, client_id: str) -> Tuple[xgb.core.DMatrix, xgb.core.DMatrix]:
        """Loads data for xgboost.

        Returns:
            A tuple of train_data, validation_data
        """
        pass
