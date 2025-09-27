# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from abc import abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class ModelAggregator(Aggregator):
    """
    Abstract class for aggregating FLModels.
    Subclasses need to implement accept_model and aggregate_model methods.
    """

    def __init__(self):
        self.fl_ctx = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.fl_ctx = fl_ctx

    @abstractmethod
    def accept_model(self, model: FLModel):
        """needs to implement logic to accept a model, e.g. add to sum, count, etc."""
        raise NotImplementedError("Subclass must implement accept_model method")

    @abstractmethod
    def aggregate_model(self) -> FLModel:
        """needs to implement aggregation logic and reset any internal stats"""
        raise NotImplementedError("Subclass must implement aggregate_model method")

    @abstractmethod
    def reset_stats(self):
        """needs to implement logic to reset any internal stats"""
        raise NotImplementedError("Subclass must implement reset_stats method")

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """called by ScatterAndGather"""
        self.fl_ctx = fl_ctx
        self.accept_model(FLModelUtils.from_shareable(shareable, fl_ctx))
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """called by ScatterAndGather"""
        self.fl_ctx = fl_ctx

        aggregated_model = self.aggregate_model()

        return FLModelUtils.to_shareable(aggregated_model)

    def reset(self, fl_ctx: FLContext):
        """called by ScatterAndGather"""
        self.fl_ctx = fl_ctx
        self.reset_stats()

    def info(self, message: str):
        self.log_info(fl_ctx=self.fl_ctx, msg=message)

    def warning(self, message: str):
        self.log_warning(fl_ctx=self.fl_ctx, msg=message)

    def error(self, message: str):
        self.log_error(fl_ctx=self.fl_ctx, msg=message)

    def exception(self, message: str):
        self.log_exception(fl_ctx=self.fl_ctx, msg=message)
