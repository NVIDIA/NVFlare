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
import time

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class FileTransferController(Controller):

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, f"Entering control loop of {self.__class__.__name__}")
            engine = fl_ctx.get_engine()

            # Wait till receiver is done. Otherwise, the job ends.
            receiver = engine.get_component("receiver")
            while not receiver.is_done():
                time.sleep(0.2)

            # Wait for a while to make sure the reply is sent to client
            time.sleep(5)
            self.log_info(fl_ctx, "Control flow ends")
        except Exception as ex:
            self.log_error(fl_ctx, f"Control flow error: {ex}")

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Start controller")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stop controller")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        raise RuntimeError(f"Unknown task: {task_name} from client {client.name}.")
