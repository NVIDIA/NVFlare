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
import random
import threading
import time
from typing import Optional

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.edge.aggregators.num_dxo import NumDXOAggregator
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.edge.mud import BaseState, ModelUpdate, StateUpdateReply, StateUpdateReport


class _ModelState:

    def __init__(self, aggr: NumDXOAggregator):
        self.aggregator = aggr
        self.devices = {}
        self.last_update_time = None

    def accept(self, model_update: ModelUpdate, fl_ctx: FLContext):
        self.last_update_time = time.time()
        self.devices.update(model_update.devices)
        return self.aggregator.accept(model_update.update, fl_ctx)


class AsyncNumAssessor(Assessor):

    def __init__(
        self,
        num_updates_for_model,
        max_model_version,
        max_model_history,
        device_selection_size,
        min_hole_to_fill=1,
        device_reuse=True,
    ):
        Assessor.__init__(self)
        self.current_model_version = 0
        self.current_model = None
        self.current_selection_version = 0
        self.current_selection = {}
        self.updates = {}  # model_version => _ModelState
        self.available_devices = {}
        self.used_devices = {}
        self.num_updates_for_model = num_updates_for_model
        self.max_model_version = max_model_version
        self.max_model_history = max_model_history
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.update_lock = threading.Lock()
        self.start_time = None

    def start_task(self, fl_ctx: FLContext) -> Shareable:
        self.start_time = time.time()
        base_state = BaseState(
            model_version=self.current_model_version,
            model=self.current_model,
            device_selection_version=self.current_selection_version,
            device_selection=self.current_selection,
        )
        return base_state.to_shareable()

    def _generate_new_model(self, fl_ctx: FLContext):
        total = 0.0
        self.current_model_version += 1
        old_model_versions = []
        aggr_info = {}
        for v, ms in self.updates.items():
            weight = 1 / (self.current_model_version - v)
            assert isinstance(ms, _ModelState)
            aggr = ms.aggregator
            assert isinstance(aggr, NumDXOAggregator)
            score = aggr.value / aggr.count if aggr.count > 0 else 0.0
            aggr_info[v] = {"weight": weight, "value": aggr.value, "count": aggr.count, "score": score}
            total += weight * score

            if self.current_model_version - v >= self.max_model_history:
                old_model_versions.append(v)

        # create the ModelState for the new model version
        self.updates[self.current_model_version] = _ModelState(NumDXOAggregator())
        self.log_info(fl_ctx, f"model version info: {aggr_info}")
        self.log_info(fl_ctx, f"generated new model version {self.current_model_version}: value={total}")

        for v in old_model_versions:
            self.updates.pop(v)

        if old_model_versions:
            self.log_info(fl_ctx, f"removed old model versions {old_model_versions}")

        self.current_model = DXO(data_kind="number", data={"value": total})

    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        with self.update_lock:
            return self._do_child_update(update, fl_ctx)

    def _do_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        report = StateUpdateReport.from_shareable(update)
        if report.available_devices:
            self.available_devices.update(report.available_devices)
            self.log_debug(
                fl_ctx,
                f"assessor got reported {len(report.available_devices)} available devices from child. "
                f"total num available devices: {len(self.available_devices)}",
            )

        accepted = True
        if report.model_updates:
            self.log_info(fl_ctx, f"got reported {len(report.model_updates)} model versions")
            for model_version, model_update in report.model_updates.items():
                if model_version <= 0:
                    continue

                if not model_update:
                    self.log_error(fl_ctx, f"bad child update version {model_version}: no update data")
                    continue

                if self.current_model_version - model_version > self.max_model_history:
                    # this version is too old
                    self.log_info(
                        fl_ctx,
                        f"dropped child update version {model_version}. Current version {self.current_model_version}",
                    )
                    continue

                model_state = self.updates.get(model_version)
                if not model_state:
                    self.log_error(fl_ctx, f"No model state for version {model_version}")
                    continue

                accepted = model_state.accept(model_update, fl_ctx)
                self.log_info(
                    fl_ctx,
                    f"processed child update V{model_version} with {len(model_update.devices)} devices: {accepted=}",
                )

                # remove reported devices from selection, and from used devices if device_reuse is enabled
                # indicating that the reported devices becomes available again for reuse
                for k in model_update.devices.keys():
                    if k not in self.current_selection:
                        self.log_error(
                            fl_ctx,
                            f"got update from device {k} but it's not in device selection",
                        )
                    self.current_selection.pop(k, None)
                    if self.device_reuse:
                        self.used_devices.pop(k, None)

            current_model_state = self.updates.get(self.current_model_version)
            if not isinstance(current_model_state, _ModelState):
                self.log_error(
                    fl_ctx, f"bad model state for version {self.current_model_version}: {type(current_model_state)}"
                )
            else:
                num_updates = len(current_model_state.devices)
                if num_updates >= self.num_updates_for_model:
                    self.log_info(
                        fl_ctx,
                        f"model V{self.current_model_version} got {num_updates} updates: generate new model version",
                    )
                    self._generate_new_model(fl_ctx)

            # recompute selection
            num_holes = self.device_selection_size - len(self.current_selection)
            if num_holes >= self.min_hole_to_fill:
                self._fill_selection(fl_ctx)
        else:
            self.log_debug(fl_ctx, "no model updates")

        # reply
        if self.current_model_version == 0:
            # do we have enough devices?
            if len(self.available_devices) >= self.device_selection_size:
                self.log_info(fl_ctx, f"got {len(self.available_devices)} devices - generate initial model")
                self._generate_new_model(fl_ctx)
                self._fill_selection(fl_ctx)

        model = None
        if self.current_model_version != report.current_model_version:
            model = self.current_model

        reply = StateUpdateReply(
            model_version=self.current_model_version,
            model=model,
            device_selection_version=self.current_selection_version,
            device_selection=self.current_selection,
        )
        return accepted, reply.to_shareable()

    def _fill_selection(self, fl_ctx: FLContext):
        num_holes = self.device_selection_size - len(self.current_selection)
        self.log_info(fl_ctx, f"filling {num_holes} holes in selection list")
        if num_holes > 0:
            self.current_selection_version += 1
            # remove all used devices from available devices
            usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())
            if usable_devices:
                for _ in range(num_holes):
                    device_id = random.choice(list(usable_devices))
                    usable_devices.remove(device_id)
                    self.current_selection[device_id] = self.current_model_version
                    self.used_devices[device_id] = {
                        "model_version": self.current_model_version,
                        "selection_version": self.current_selection_version,
                    }
                    if not usable_devices:
                        break
        self.log_info(
            fl_ctx,
            f"current selection with {len(self.current_selection)} items: V{self.current_selection_version}; {dict(sorted(self.current_selection.items()))}",
        )
        if len(self.current_selection) < self.device_selection_size:
            self.log_warning(
                fl_ctx,
                f"current selection has only {len(self.current_selection)} devices, which is less than the expected {self.device_selection_size} devices. Please check the configuration to make sure this is expected.",
            )

    def assess(self, fl_ctx: FLContext) -> Assessment:
        if self.current_model_version >= self.max_model_version:
            model_version = self.current_model_version
            selection_version = self.current_selection_version
            self.log_info(
                fl_ctx,
                f"Max model version {self.max_model_version} reached: {model_version=} {selection_version=} "
                f"num of devices used: {len(self.used_devices)}",
            )
            return Assessment.WORKFLOW_DONE
        else:
            return Assessment.CONTINUE
