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
import copy
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.mud import BaseState, DeviceInfo, ModelUpdate, StateUpdateReply, StateUpdateReport
from nvflare.edge.updater import Updater


class AggregatorFactory(ABC):

    @abstractmethod
    def get_aggregator(self) -> Aggregator:
        pass


class ModelAggrState:

    def __init__(self, aggregator: Aggregator, model_version: int):
        self.aggregator = aggregator
        self.model_version = model_version
        self.devices: Dict[str, float] = {}

    def accept(self, contribution: Shareable, devices: Dict[str, float], fl_ctx: FLContext) -> bool:
        accepted = self.aggregator.accept(contribution, fl_ctx)
        if accepted:
            self.devices.update(devices)
        return accepted

    def to_model_update(self, fl_ctx: FLContext) -> ModelUpdate:
        return ModelUpdate(
            model_version=self.model_version,
            update=self.aggregator.aggregate(fl_ctx),
            devices=self.devices,
        )

    def reset(self, fl_ctx: FLContext):
        self.aggregator.reset(fl_ctx)
        self.devices = {}


class EdgeModelUpdater(Updater):

    def __init__(self, aggr_factory_id: Union[str, AggregatorFactory], max_model_versions: int):
        Updater.__init__(self)
        self.aggr_factory_id = aggr_factory_id
        self.max_model_versions = max_model_versions
        self.aggr_factory = None
        self.aggregators: List[ModelAggrState] = []
        self.available_devices: Dict[str, DeviceInfo] = {}  # device_id => Device
        self._update_lock = threading.Lock()
        self.register_event_handler(EventType.START_RUN, self._emu_handle_start_run)

        if isinstance(aggr_factory_id, AggregatorFactory):
            self.aggr_factory = aggr_factory_id

    def _emu_handle_start_run(self, event_type: str, fl_ctx: FLContext):
        if not isinstance(self.aggr_factory_id, AggregatorFactory):
            engine = fl_ctx.get_engine()
            factory = engine.get_component(self.aggr_factory_id)
            if not isinstance(factory, AggregatorFactory):
                self.system_panic(
                    f"Component {self.aggr_factory_id} should be AggregatorFactory but got {type(factory)}", fl_ctx
                )
                return
        else:
            factory = self.aggr_factory_id
        self.aggr_factory = factory

    def start_task(self, task_data: Shareable, fl_ctx: FLContext) -> Any:
        # The task_data is a BaseState
        self.current_state = BaseState.from_shareable(task_data)
        return self.current_state

    def prepare_update_for_parent(self, fl_ctx: FLContext) -> Optional[Shareable]:
        state = self.current_state
        assert isinstance(state, BaseState)
        with self._update_lock:
            report = StateUpdateReport(
                current_model_version=state.model_version,
                current_device_list_version=state.device_list_version,
                model_updates=[a.to_model_update(fl_ctx) for a in self.aggregators],
                devices=[d for d in self.available_devices.values()],
            )

            for a in self.aggregators:
                a.reset(fl_ctx)

            return report.to_shareable()

    def process_parent_update_reply(self, reply: Shareable, fl_ctx: FLContext):
        update_reply = StateUpdateReply.from_shareable(reply)

        # update the current_state
        num_changes = 0
        with self._update_lock:
            new_state = copy.copy(self.current_state)

        if update_reply.model_version != new_state.model_version:
            new_state.model_version = update_reply.model_version
            new_state.model = update_reply.model
            num_changes += 1

        if update_reply.device_list_version != new_state.device_list_version:
            new_state.device_list_version = update_reply.device_list_version
            new_state.device_list = update_reply.device_list
            num_changes += 1

        if num_changes > 0:
            self.current_state = new_state

        # drop old aggrs
        with self._update_lock:
            old_aggrs = []
            for a in self.aggregators:
                if new_state.model_version - a.model_version > self.max_model_versions:
                    old_aggrs.append(a)

            for a in old_aggrs:
                self.aggregators.remove(a)

    def _update_one_model(self, mu: ModelUpdate, fl_ctx: FLContext):
        mas = None
        for a in self.aggregators:
            if mu.model_version == a.model_version:
                mas = a
                break

        if not mas:
            assert isinstance(self.aggr_factory, AggregatorFactory)
            aggr = self.aggr_factory.get_aggregator()
            mas = ModelAggrState(aggr, mu.model_version)
            self.aggregators.append(mas)

        return mas.accept(mu.update, mu.devices, fl_ctx)

    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        report = StateUpdateReport.from_shareable(update)
        with self._update_lock:
            if report.devices:
                for d in report.devices:
                    self.available_devices[d.device_id] = d

            if report.model_updates:
                for mu in report.model_updates:
                    self._update_one_model(mu, fl_ctx)

            if report.current_model_version is None:
                # local report
                return True, None

            # send base state back
            state = self.current_state
            model = state.model if state.model_version != report.current_model_version else None
            dev_list = state.device_list
            if state.device_list_version == report.current_device_list_version:
                dev_list = None

            reply = StateUpdateReply(
                model_version=state.model_version,
                model=model,
                device_list_version=state.device_list_version,
                device_list=dev_list,
            )

            return True, reply.to_shareable()

    def end_task(self, fl_ctx: FLContext):
        super().end_task(fl_ctx)
        for a in self.aggregators:
            a.reset(fl_ctx)
