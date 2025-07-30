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
from abc import ABC, abstractmethod
from typing import Any

from nvflare.apis.dxo import DXO
from nvflare.apis.signal import Signal


class Context(dict):

    def fire_event(self, event_type: str, data: Any, abort_signal: Signal):
        handlers = self.get(ContextKey.EVENT_HANDLERS)
        if handlers:
            for h in handlers:
                h.handle_event(event_type, data, self, abort_signal)


class ContextKey:
    RUNNER = "runner"
    DATA_SOURCE = "data_source"
    EXECUTOR = "executor"
    COMPONENTS = "components"
    EVENT_HANDLERS = "event_handlers"
    TASK_NAME = "task_name"
    TASK_ID = "task_id"
    TASK_DATA = "task_data"


class EventType:
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    LOSS_GENERATED = "loss_generated"


class Batch:

    def get_input(self) -> Any:
        pass

    def get_label(self) -> Any:
        pass


class Dataset(ABC):

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def get_next_batch(self, batch_size) -> Batch:
        pass

    @abstractmethod
    def reset(self):
        pass


class DataSource(ABC):

    @abstractmethod
    def get_dataset(self, dataset_type: str, ctx: Context) -> Dataset:
        pass


class Executor(ABC):

    @abstractmethod
    def execute(self, task_data: DXO, ctx: Context, abort_signal: Signal) -> DXO:
        pass


class Filter(ABC):

    @abstractmethod
    def filter(self, data: DXO, ctx: Context, abort_signal: Signal) -> DXO:
        pass


class Transform(ABC):

    @abstractmethod
    def transform(self, batch: Batch, ctx: Context, abort_signal: Signal) -> Batch:
        pass


class EventHandler(ABC):

    @abstractmethod
    def handle_event(self, event_type: str, event_data: Any, ctx: Context, abort_signal: Signal):
        pass
