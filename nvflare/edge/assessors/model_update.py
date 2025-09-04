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
from concurrent.futures import ThreadPoolExecutor
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
        device_wait_timeout: float = 30.0,
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
            device_wait_timeout (float): Timeout in seconds for waiting for sufficient devices. Default is 30 seconds.
        """
        Assessor.__init__(self)
        self.persistor_id = persistor_id
        self.model_manager_id = model_manager_id
        self.device_manager_id = device_manager_id
        self.persistor = None
        self.model_manager = None
        self.device_manager = None
        self.max_model_version = max_model_version
        self.device_wait_timeout = device_wait_timeout
        self.update_lock = threading.Lock()
        self.device_wait_start_time = None
        self.should_stop_job = False
        self.timeout_future = None
        self.thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DeviceTimeout")
        self.timeout_wait_event = threading.Event()
        self.timeout_thread_active = False
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _check_device_timeout(self, fl_ctx: FLContext) -> bool:
        """Check if device wait timeout has been exceeded.

        Args:
            fl_ctx: FL context

        Returns:
            bool: True if timeout exceeded, False otherwise
        """
        if self.device_wait_start_time is None:
            return False

        if time.time() - self.device_wait_start_time > self.device_wait_timeout:
            return True
        return False

    def _log_device_wait_status(self, fl_ctx: FLContext, message_prefix: str = ""):
        """Log current device wait status with countdown information."""
        if self.device_wait_start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.device_wait_start_time

            # Only log if we haven't logged recently (rate limiting)
            if not hasattr(self, "_last_status_log_time"):
                self._last_status_log_time = 0

            # Adaptive logging frequency based on urgency
            remaining_time = self.device_wait_timeout - elapsed
            if remaining_time > 0:
                # Determine logging interval based on remaining time
                if remaining_time > 60:  # More than 1 minute: log every 30 seconds
                    log_interval = 30.0
                elif remaining_time > 30:  # 30 seconds to 1 minute: log every 15 seconds
                    log_interval = 15.0
                elif remaining_time > 10:  # 10 to 30 seconds: log every 5 seconds
                    log_interval = 5.0
                else:  # Final 10 seconds: log every 2 seconds
                    log_interval = 2.0

                # Check if enough time has passed since last log
                if current_time - self._last_status_log_time >= log_interval:
                    usable_devices = set(self.device_manager.available_devices.keys()) - set(
                        self.device_manager.used_devices.keys()
                    )
                    self.log_info(
                        fl_ctx,
                        f"{message_prefix}Device wait status: "
                        f"Total devices: {len(self.device_manager.available_devices)}, "
                        f"usable: {len(usable_devices)}, "
                        f"expected: {self.device_manager.device_selection_size}. "
                        f"Timeout in {remaining_time:.1f} seconds.",
                    )
                    self._last_status_log_time = current_time

    def _start_timeout_tracking(self, fl_ctx: FLContext):
        """Start timeout tracking using long-lived thread."""
        if not self.timeout_thread_active:
            # Start the long-lived timeout thread if not already running
            self.timeout_future = self.thread_pool.submit(self._timeout_tracker, fl_ctx)
            self.timeout_thread_active = True
            self.log_debug(fl_ctx, f"Started device wait timeout tracking for {self.device_wait_timeout}s")

        # Signal the thread to start monitoring
        self.timeout_wait_event.set()

    def _stop_timeout_tracking(self):
        """Stop timeout tracking."""
        # Clear the wait event to signal the thread to stop monitoring
        self.timeout_wait_event.clear()

    def _timeout_tracker(self, fl_ctx: FLContext):
        """Long-lived timeout tracker that runs in thread pool."""
        try:
            while True:
                # Wait for the event to be set (indicating we should start monitoring)
                if not self.timeout_wait_event.wait(timeout=1.0):
                    # Timeout on wait - check if we should exit the thread
                    if self.should_stop_job:
                        break
                    continue

                # Event was set, start monitoring timeout
                if self.device_wait_start_time is None:
                    # No timeout to monitor, clear event and continue
                    self.timeout_wait_event.clear()
                    continue

                # Periodic logging during countdown to keep users informed
                check_interval = 10.0  # Log every 10 seconds
                elapsed = 0

                while (
                    self.device_wait_start_time is not None
                    and not self.should_stop_job
                    and elapsed < self.device_wait_timeout
                    and self.timeout_wait_event.is_set()
                ):
                    time.sleep(min(check_interval, self.device_wait_timeout - elapsed))
                    elapsed += check_interval

                    # Check if we should stop monitoring
                    if (
                        self.device_wait_start_time is None
                        or self.should_stop_job
                        or not self.timeout_wait_event.is_set()
                    ):
                        break

                    # Log periodic status update
                    if elapsed < self.device_wait_timeout:
                        remaining = self.device_wait_timeout - elapsed
                        self.log_info(fl_ctx, f"Device wait countdown: {remaining:.1f} seconds remaining")

                # Check if we're still waiting for devices and timeout exceeded
                if (
                    self.device_wait_start_time is not None
                    and not self.should_stop_job
                    and self._check_device_timeout(fl_ctx)
                    and self.timeout_wait_event.is_set()
                ):
                    # Timeout exceeded, set the stop flag
                    self.should_stop_job = True
                    self.log_error(
                        fl_ctx,
                        f"Device wait timeout ({self.device_wait_timeout}s) exceeded. "
                        f"Setting stop job flag to terminate workflow.",
                    )

                # Clear the event to stop monitoring
                self.timeout_wait_event.clear()

        except Exception as e:
            # Log error but don't crash the thread
            self.log_error(fl_ctx, f"Error in timeout tracker: {e}")
        finally:
            # Mark thread as inactive
            self.timeout_thread_active = False

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
            # Reset wait timer if we now have enough devices
            if self.device_wait_start_time is not None and self.device_manager.has_enough_devices(fl_ctx):
                self.device_wait_start_time = None
                self.should_stop_job = False  # Reset stop job flag
                # Stop timeout tracking since we have enough devices
                self._stop_timeout_tracking()
                self.log_info(fl_ctx, "Sufficient devices now available, resetting wait timer and stop job flag")

        # Check for device wait timeout if we are waiting for devices
        if self.device_wait_start_time is not None:
            if self.should_stop_job:
                # Timeout exceeded, prepare an empty reply
                usable_devices = set(self.device_manager.available_devices.keys()) - set(
                    self.device_manager.used_devices.keys()
                )
                self.log_error(
                    fl_ctx,
                    f"Device wait timeout ({self.device_wait_timeout}s) exceeded. "
                    f"Total devices: {len(self.device_manager.available_devices)}, usable: {len(usable_devices)}, expected: {self.device_manager.device_selection_size}. "
                    f"Device_reuse flag is set to: {self.device_manager.device_reuse}. "
                    "Not enough devices joining, please adjust the server params. Stopping the job.",
                )
                reply = StateUpdateReply(
                    model_version=0,
                    model=None,
                    device_selection_version=self.device_manager.current_selection_version,
                    device_selection=self.device_manager.get_selection(fl_ctx),
                )
                return False, reply.to_shareable()
            else:
                # Log current wait status for user information (rate limited)
                self._log_device_wait_status(fl_ctx, "Waiting for devices: ")

        accepted = True
        if report.model_updates:
            self.log_info(fl_ctx, f"got reported {len(report.model_updates)} model versions")

            # Process model updates
            accepted = self.model_manager.process_updates(report.model_updates, fl_ctx)

            # Remove reported devices from selection
            for model_update in report.model_updates.values():
                if model_update:
                    self.device_manager.remove_devices_from_selection(set(model_update.devices.keys()), fl_ctx)
                    # if device_reuse, remove devices from used_devices
                    # indicating that the reported devices becomes available again for reuse
                    if self.device_manager.device_reuse:
                        self.device_manager.remove_devices_from_used(set(model_update.devices.keys()), fl_ctx)

        else:
            self.log_debug(fl_ctx, "no model updates")

        # Handle device selection
        if self.device_manager.should_fill_selection(fl_ctx):
            # check if we have enough devices to fill selection
            if self.device_manager.has_enough_devices(fl_ctx):
                if self.model_manager.current_model_version == 0:
                    self.log_info(fl_ctx, "Generate initial model and fill selection")
                    self.model_manager.generate_new_model(fl_ctx)
                self.device_manager.fill_selection(self.model_manager.current_model_version, fl_ctx)
                # Reset wait timer since we have enough devices
                self.device_wait_start_time = None
                # Stop timeout tracking since we have enough devices
                self._stop_timeout_tracking()
            else:
                # Start or continue wait timer since we don't have enough devices
                if self.device_wait_start_time is None:
                    self.device_wait_start_time = time.time()
                    # Start independent timeout tracking
                    self._start_timeout_tracking(fl_ctx)

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
        if self.should_stop_job:
            # Stop timeout tracking before ending the job
            self._stop_timeout_tracking()
            self.log_error(fl_ctx, "Job stopped due to insufficient devices joining within timeout period")
            return Assessment.WORKFLOW_DONE
        elif self.model_manager.current_model_version >= self.max_model_version:
            # Stop timeout tracking before ending the job
            self._stop_timeout_tracking()
            model_version = self.model_manager.current_model_version
            self.log_info(fl_ctx, f"Max model version {self.max_model_version} reached: {model_version=}")
            return Assessment.WORKFLOW_DONE
        else:
            return Assessment.CONTINUE

    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, "thread_pool"):
            # Signal the timeout thread to exit
            self.should_stop_job = True
            if hasattr(self, "timeout_wait_event"):
                self.timeout_wait_event.set()
            self.thread_pool.shutdown(wait=False)
