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

# We will move to this app_common when it gets matured
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.sklearn.data_loader import load_data_for_range


class SKLearner(FLComponent, ABC):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
    ):
        self.fl_ctx = None
        self.data_path = data_path
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        super().__init__()

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    def load_data(self) -> dict:
        train_data = load_data_for_range(self.data_path, self.train_start, self.train_end)
        valid_data = load_data_for_range(self.data_path, self.valid_start, self.valid_end)
        return {"train": train_data, "valid": valid_data}

    def get_parameters(self, global_param: Optional[dict] = None) -> dict:
        pass

    @abstractmethod
    def train(self, curr_round: int, global_param: Optional[dict] = None) -> Tuple[dict, dict]:
        pass

    @abstractmethod
    def evaluate(self, curr_round: int, global_param: Optional[dict] = None) -> Tuple[dict, dict]:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass
