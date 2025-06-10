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

import threading
import time
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, model_learnable_to_dxo
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.edge.mud import BaseState, StateUpdateReply, StateUpdateReport


class ModelUpdateAssessor(Assessor):
    def __init__(
        self,
        persistor_id,
        model_manager_id,
        device_manager_id,
        max_model_version,
    ):
        """Initialize the ModelUpdateAssessor.
        Enable both asynchronous and synchronous model updates from clients.
        For asynchronous updates, the staleness is calculated based on the starting and current model version.
        And the aggregation scheme and weights are calculated following FedBuff paper "Federated Learning with Buffered Asynchronous Aggregation".

        Args:
            persistor_id (str): ID of the persistor component used to load and save models.
            model_manager_id (str): ID of the model manager component.
            device_manager_id (str): ID of the device manager component.
            max_model_version (int): Maximum model version to stop the workflow.
        """
        Assessor.__init__(self)
        self.persistor_id = persistor_id
        self.model_manager_id = model_manager_id
        self.device_manager_id = device_manager_id
        self.persistor = None
        self.model_manager = None
        self.device_manager = None
        self.max_model_version = max_model_version
        self.update_lock = threading.Lock()
        self.start_time = None
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        # Get persistor component
        self.persistor = engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(reason="persistor must be a Persistor type object", fl_ctx=fl_ctx)
            return

        # Get model manager component
        self.model_manager = engine.get_component(self.model_manager_id)
        if not self.model_manager:
            self.system_panic(reason=f"cannot find model manager component '{self.model_manager_id}'", fl_ctx=fl_ctx)
            return

        # Get device manager component
        self.device_manager = engine.get_component(self.device_manager_id)
        if not self.device_manager:
            self.system_panic(reason=f"cannot find device manager component '{self.device_manager_id}'", fl_ctx=fl_ctx)
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
            dxo_model = model_learnable_to_dxo(model)
            self.model_manager.initialize_model(dxo_model, fl_ctx)
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

    def process_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        with self.update_lock:
            return self._do_child_update(update, fl_ctx)

    def _do_child_update(self, update: Shareable, fl_ctx: FLContext) -> (bool, Optional[Shareable]):
        report = StateUpdateReport.from_shareable(update)

        # Update available devices
        if report.available_devices:
            self.device_manager.update_available_devices(report.available_devices, fl_ctx)

        accepted = True
        if report.model_updates:
            self.log_info(fl_ctx, f"got reported {len(report.model_updates)} model versions")

            # Process model updates
            accepted = self.model_manager.process_updates(report.model_updates, fl_ctx)

            # Remove reported devices from selection
            for model_update in report.model_updates.values():
                if model_update:
                    self.device_manager.remove_devices_from_selection(set(model_update.devices.keys()), fl_ctx)

            # Handle device selection
            if self.device_manager.should_fill_selection(fl_ctx):
                self.device_manager.fill_selection(self.model_manager.current_model_version, fl_ctx)

        else:
            self.log_debug(fl_ctx, "no model updates")

        # Handle initial model generation
        if self.model_manager.current_model_version == 0:
            if self.device_manager.has_enough_devices(fl_ctx):
                self.log_info(
                    fl_ctx, f"got {len(self.device_manager.available_devices)} devices - generate initial model"
                )
                self.model_manager.generate_new_model(fl_ctx)
                self.device_manager.fill_selection(self.model_manager.current_model_version, fl_ctx)

        # Prepare reply
        model = None
        if self.model_manager.current_model_version != report.current_model_version:
            model = self.model_manager.get_current_model(fl_ctx)

        reply = StateUpdateReply(
            model_version=self.model_manager.current_model_version,
            model=model,
            device_selection_version=self.device_manager.current_selection_version,
            device_selection=self.device_manager.get_selection(fl_ctx),
        )
        return accepted, reply.to_shareable()

    def assess(self, fl_ctx: FLContext) -> Assessment:
        if self.model_manager.current_model_version >= self.max_model_version:
            model_version = self.model_manager.current_model_version
            self.log_info(fl_ctx, f"Max model version {self.max_model_version} reached: {model_version=}")
            return Assessment.WORKFLOW_DONE
        else:
            return Assessment.CONTINUE
