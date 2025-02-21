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
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class EdgeSurveyController(Controller):
    def __init__(self, num_rounds: int, timeout: int):
        Controller.__init__(self)
        self.num_rounds = num_rounds
        self.timeout = timeout

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        for r in range(self.num_rounds):
            task = Task(
                name="survey",
                data=Shareable(),
                timeout=self.timeout,
            )

            self.broadcast_and_wait(
                task=task,
                min_responses=2,
                wait_time_after_min_received=0,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            total_devices = 0
            for ct in task.client_tasks:
                assert isinstance(ct, ClientTask)
                result = ct.result
                assert isinstance(result, Shareable)
                self.log_info(fl_ctx, f"result from client {ct.client.name}: {result}")
                count = result.get("num_devices")
                if count:
                    total_devices += count

            self.log_info(fl_ctx, f"total devices in round {r}: {total_devices}")
