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
from abc import ABC
from typing import Dict

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class Assembler(FLComponent, ABC):
    """
    Collector is special aggregator
    Collector is responsible for the communication with clients and accept contributions
    Collector delegate the aggregation to Assembler
    """

    def __init__(self, data_kind: str):
        super().__init__()
        self.fl_ctx = None
        self.expected_data_kind = data_kind
        self.logger.debug(f"expected data kind: {self.expected_data_kind}")
        self.accumulator: dict = {}

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    def get_accumulator(self):
        return self.accumulator

    def get_expected_data_kind(self):
        return self.expected_data_kind

    def get_model_params(self, data: dict) -> dict:
        pass

    def aggregate(self, current_round: int, data: Dict[str, dict]) -> dict:
        pass
