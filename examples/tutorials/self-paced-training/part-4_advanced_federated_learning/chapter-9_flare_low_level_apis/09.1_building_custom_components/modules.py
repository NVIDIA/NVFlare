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
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class PlaceholderController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass


class PlaceholderExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        pass


class LoggingController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")


class LoggingExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        pass

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            self.log_info(fl_ctx, "Starting the executor...")
        elif event_type == EventType.END_RUN:
            self.log_info(fl_ctx, "Stopping the executor...")
