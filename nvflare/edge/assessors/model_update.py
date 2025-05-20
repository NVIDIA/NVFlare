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

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.aggregators.model_update_dxo import ModelUpdateDXOAggregator
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.edge.mud import BaseState, ModelUpdate, StateUpdateReply, StateUpdateReport


class _ModelState:

    def __init__(self, aggr: ModelUpdateDXOAggregator):
        self.aggregator = aggr
        self.devices = {}
        self.last_update_time = None

    def accept(self, model_update: ModelUpdate, fl_ctx: FLContext):
        self.last_update_time = time.time()
        self.devices.update(model_update.devices)
        return self.aggregator.accept(model_update.update, fl_ctx)


class ModelUpdateAssessor(Assessor):

    def __init__(
        self,
        persistor_id,
        num_updates_for_model,
        max_model_version,
        max_model_history,
        device_selection_size,
        min_hole_to_fill=1,
        device_reuse=True,
    ):
        Assessor.__init__(self)
        self.persistor_id = persistor_id
        self.persistor = None
        self.current_model = None
        self.current_model_version = 0
        self.current_selection = {}
        self.current_selection_version = 0
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
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.persistor = engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(reason="persistor must be a Persistor type object", fl_ctx=fl_ctx)
            return
        if self.persistor:
            model = self.persistor.load(fl_ctx)

            if not isinstance(model, ModelLearnable):
                self.system_panic(
                    reason=f"Expected model loaded by persistor to be `ModelLearnable` but received {type(model)}",
                    fl_ctx=fl_ctx,
                )
                return

            # Wrap learnable model into a DXO
            self.current_model = model_learnable_to_dxo(model)
            self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)
        else:
            self.system_panic(reason="cannot find persistor component '{}'".format(self.persistor_id), fl_ctx=fl_ctx)
            return

    def start_task(self, fl_ctx: FLContext) -> Shareable:
        self.start_time = time.time()
        # empty base state to start with
        base_state = BaseState(
            model_version=0,
            model=None,
            device_selection_version=0,
            device_selection={},
        )
        return base_state.to_shareable()

    def _generate_new_model(self, fl_ctx: FLContext):
        # New model generated based on the current global weights and all updates
        new_model = {}
        self.current_model_version += 1
        old_model_versions = []

        if self.current_model_version == 1:
            # Initial global weights
            new_model = self.current_model.data
        else:
            # Aggregate all updates
            for v, ms in self.updates.items():
                # FedBuff weight is 1/(1+staleness)^0.5
                weight = 1 / (1 + (self.current_model_version - v) ** 0.5)
                assert isinstance(ms, _ModelState)
                aggr = ms.aggregator
                assert isinstance(aggr, ModelUpdateDXOAggregator)
                # Add the dict to new_model by multiplying the weight and dividing by the count
                update_dict = aggr.dict
                count = aggr.count
                if count > 0:
                    # aggregate updates
                    for key, value in update_dict.items():
                        # apply weight and divide by count
                        value = weight * value / count
                        # update the new model
                        if key not in new_model:
                            new_model[key] = value
                        else:
                            new_model[key] = new_model[key] + value
                # If too old, remove it
                if self.current_model_version - v >= self.max_model_history:
                    old_model_versions.append(v)

            # Add the aggregated updates to the current global weights
            global_weights = self.current_model.data
            for key, value in new_model.items():
                # check key alignment
                if key not in global_weights:
                    self.log_error(fl_ctx, f"key {key} not in new model")
                    continue
                else:
                    new_model[key] = np.array(global_weights[key]) + value

        # create the ModelState for the new model version
        self.updates[self.current_model_version] = _ModelState(ModelUpdateDXOAggregator())
        self.log_info(fl_ctx, f"generated new model version {self.current_model_version}")

        for v in old_model_versions:
            self.updates.pop(v)

        if old_model_versions:
            self.log_info(fl_ctx, f"removed old model versions {old_model_versions}")

        # update the current model
        self.current_model = DXO(data_kind=DataKind.WEIGHTS, data=new_model)

        # set fl_ctx and fire the event
        # wrap new_model to a learnable
        learnable = make_model_learnable(new_model, {})
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learnable, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_model_version, private=True, sticky=True)
        self.fire_event(AppEventType.GLOBAL_WEIGHTS_UPDATED, fl_ctx)

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

                # remove reported devices from selection
                for k in model_update.devices.keys():
                    if k not in self.current_selection:
                        self.log_error(fl_ctx, f"got update from device {k} but it's not in device selection")
                    self.current_selection.pop(k, None)

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
            if not self.device_reuse:
                # remove all used devices from available devices
                usable_devices = set(self.available_devices.keys()) - set(self.used_devices.keys())
            else:
                # remove only the devices that are associated with the current model version
                usable_devices = set(self.available_devices.keys()) - set(
                    k for k, v in self.used_devices.items() if v == self.current_model_version
                )

            if usable_devices:
                for _ in range(num_holes):
                    device_id = random.choice(list(usable_devices))
                    usable_devices.remove(device_id)
                    self.current_selection[device_id] = self.current_selection_version
                    self.used_devices[device_id] = self.current_model_version
                    if not usable_devices:
                        break
        self.log_info(
            fl_ctx,
            f"current selection: V{self.current_selection_version}; {dict(sorted(self.current_selection.items()))}",
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
