# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, Optional, Union

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.mud import BaseState, Device, ModelUpdate, StateUpdateReply, StateUpdateReport
from nvflare.edge.updater import Updater


class AggregatorFactory(ABC):

    # Opt in only for workflows allowing one contribution per device per model.
    unique_devices_per_model = False

    @abstractmethod
    def get_aggregator(self) -> Aggregator:
        pass


class ModelAggrState:

    def __init__(self, aggregator: Aggregator, model_version: int):
        self.aggregator = aggregator
        self.model_version = model_version
        self.devices: Dict[str, float] = {}

    def accept(self, contribution: Shareable, devices: Dict[str, float], fl_ctx: FLContext) -> bool:
        if not devices:
            raise ValueError("cannot accept contribution with no devices")

        accepted = self.aggregator.accept(contribution, fl_ctx)
        if accepted:
            self.devices.update(devices)
        else:
            raise RuntimeError(f"aggregator {type(self.aggregator)} failed to accept")
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
        self.aggr_states: Dict[int, ModelAggrState] = {}  # model_version => ModelAggrState
        # Keep identity history for the task, independently of pending reports and
        # aggregator pruning: late results can recreate a pruned model bucket.
        self._seen_devices: Dict[int, set] = {}
        self.available_devices: Dict[str, Device] = {}  # device_id => Device
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
            model_updates = {}
            for k, v in self.aggr_states.items():
                if not v.devices:
                    # nothing to report
                    continue
                model_updates[k] = v.to_model_update(fl_ctx)

            if model_updates:
                self.log_debug(fl_ctx, f"prepared {len(model_updates)} model updates for parent")

            report = StateUpdateReport(
                current_model_version=state.model_version,
                current_device_selection_version=state.device_selection_version,
                model_updates=model_updates,
                available_devices=self.available_devices,
            )

            self.log_debug(
                fl_ctx,
                f"prepared parent update report: {report.current_model_version=} "
                f"model_updates={report.model_updates.keys()}"
                f"{report.current_device_selection_version=} "
                f"available_devices={len(report.available_devices)}",
            )

            for a in self.aggr_states.values():
                a.reset(fl_ctx)

            return report.to_shareable()

    def process_parent_update_reply(self, reply: Shareable, fl_ctx: FLContext):
        update_reply = StateUpdateReply.from_shareable(reply)

        # update the current_state
        num_changes = 0
        with self._update_lock:
            # make a new state based on current state.
            # then make changes to the new state, only when necessary
            new_state = copy.copy(self.current_state)

        if update_reply.model_version != new_state.model_version:
            # model has changed.
            new_state.model_version = update_reply.model_version
            new_state.model = update_reply.model
            new_state.converted_models = {}
            num_changes += 1

        if update_reply.device_selection_version != new_state.device_selection_version:
            # device selection has changed.
            new_state.device_selection_version = update_reply.device_selection_version
            new_state.device_selection = update_reply.device_selection
            num_changes += 1

        if num_changes > 0:
            # switch to the new state in one atomic operation
            self.current_state = new_state

        # drop old aggr versions based on active_model_versions from parent
        with self._update_lock:
            old_versions = set()
            # make a set of current active model versions from unique value of device_selection dict
            active_model_versions = set(new_state.device_selection.values())
            for mv, state in self.aggr_states.items():
                # Results may arrive while waiting for this parent reply. Preserve them
                # until the next report forwards them, regardless of selection or age.
                if mv == new_state.model_version or state.devices:
                    continue
                # remove the model versions that are
                # - either not in device_selection values
                if mv not in active_model_versions:
                    old_versions.add(mv)
                # - or too old
                elif self.max_model_versions and new_state.model_version - mv > self.max_model_versions:
                    old_versions.add(mv)

            # remove old versions
            for mv in old_versions:
                self.aggr_states.pop(mv, None)
                self.log_info(fl_ctx, f"removed aggregator for model version {mv}")
                self.log_info(fl_ctx, f"current total number of active aggregator versions: {len(self.aggr_states)}")

    def _update_one_model(self, mu: ModelUpdate, fl_ctx: FLContext):
        assert isinstance(self.aggr_factory, AggregatorFactory)
        if self.aggr_factory.unique_devices_per_model:
            seen = self._seen_devices.get(mu.model_version, set())
            overlap = seen.intersection(mu.devices)
            if overlap:
                if overlap != mu.devices.keys():
                    raise ValueError(
                        f"partial overlap in model V{mu.model_version}: "
                        "cannot separate already accepted devices from a preaggregated update"
                    )
                return False

        mas = self.aggr_states.get(mu.model_version)

        if not mas:
            aggr = self.aggr_factory.get_aggregator()
            mas = ModelAggrState(aggr, mu.model_version)
            self.aggr_states[mu.model_version] = mas

        accepted = mas.accept(mu.update, mu.devices, fl_ctx)
        if accepted and self.aggr_factory.unique_devices_per_model:
            self._seen_devices.setdefault(mu.model_version, set()).update(mu.devices)
        self.log_info(fl_ctx, f"updated one model V{mu.model_version} with {len(mu.devices)} devices: {accepted=}")
        return accepted

    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        report = StateUpdateReport.from_shareable(update)
        with self._update_lock:
            if report.available_devices:
                self.available_devices.update(report.available_devices)

            accepted = True
            if report.model_updates:
                for mu in report.model_updates.values():
                    update_accepted = self._update_one_model(mu, fl_ctx)
                    accepted = update_accepted and accepted

            if report.current_model_version is None:
                # local report
                return accepted, None

            # send base state back
            state = self.current_state

            if not state:
                # no current state data
                return accepted, None

            assert isinstance(state, BaseState)
            model = state.model if state.model_version != report.current_model_version else None
            dev_selection = state.device_selection
            if state.device_selection_version == report.current_device_selection_version:
                dev_selection = None

            reply = StateUpdateReply(
                model_version=state.model_version,
                model=model,
                device_selection_version=state.device_selection_version,
                device_selection=dev_selection,
            )

            self.log_debug(
                fl_ctx,
                f"accepted {len(report.available_devices)} available devices from child "
                f"total available devices is now {len(self.available_devices)}",
            )

            return accepted, reply.to_shareable()

    def end_task(self, fl_ctx: FLContext):
        super().end_task(fl_ctx)
        for a in self.aggr_states.values():
            a.reset(fl_ctx)
        self.aggr_states = {}
        self._seen_devices = {}
