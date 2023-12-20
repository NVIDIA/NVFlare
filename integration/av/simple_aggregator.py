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
from typing import Any

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_event_type import AppEventType


class SimpleAggregator(Aggregator, ABC):
    def __init__(self):
        Aggregator.__init__(self)
        self.fl_ctx = None

    def handle_event(self, event_type, fl_ctx: FLContext):
        if event_type == AppEventType.ROUND_STARTED:
            self.fl_ctx = fl_ctx
            self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def processing_training_result(self, client_name: str, trained_weights: Any, trained_meta: dict) -> bool:
        pass

    @abstractmethod
    def aggregate_training_result(self) -> (Any, dict):
        pass

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        trained_weights = dxo.data
        if dxo.data_kind == DataKind.APP_DEFINED:
            trained_weights = trained_weights.get(DataKind.APP_DEFINED)
        trained_meta = dxo.meta
        self.fl_ctx = fl_ctx
        peer_ctx = fl_ctx.get_peer_context()
        client_name = peer_ctx.get_identity_name()
        return self.processing_training_result(client_name, trained_weights, trained_meta)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        self.fl_ctx = fl_ctx
        aggregated_result, aggregated_meta = self.aggregate_training_result()
        dxo = DXO(
            data_kind=DataKind.APP_DEFINED,
            data={DataKind.APP_DEFINED: aggregated_result},
            meta=aggregated_meta,
        )
        print(f"learnable_to_shareable: {dxo.data}")
        return dxo.to_shareable()
