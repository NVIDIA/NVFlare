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

from abc import ABC, abstractmethod
from typing import Dict

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class Assembler(FLComponent, ABC):
    """Assembler class for aggregation functionality
    This defines the functionality of assembling the collected submissions
    for CollectAndAssembleAggragator
    """

    def __init__(self, data_kind: str):
        super().__init__()
        self.fl_ctx = None
        self.expected_data_kind = data_kind
        self.logger.debug(f"expected data kind: {self.expected_data_kind}")
        self.collector: dict = {}

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    def get_collector(self):
        return self.collector

    def get_expected_data_kind(self):
        return self.expected_data_kind

    @abstractmethod
    def get_model_params(self, data: dict) -> dict:
        """Connects the assembler's collector with CollectAndAssembleAggregator
        Get the collected parameters from the main aggregator
        Return:
            A dict of parameters needed for further assembling
        """
        raise NotImplementedError

    @abstractmethod
    def assemble(self, current_round: int, data: Dict[str, dict]) -> dict:
        """Assemble the collected submissions.
        This will be specified according to the specific algorithm
        E.g. global svm round on the collected local supporting vectors;
        global k-means step on the local centroids and counts

        Return:
            A dict of parameters to be returned to clients
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        # Reset parameters for next round,
        # This will be performed at the end of each aggregation round,
        # it can include, but not limited to, clearing the collector
        raise NotImplementedError
