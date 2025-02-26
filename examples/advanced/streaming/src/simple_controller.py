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

import logging
import time

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

logger = logging.getLogger(__name__)


class SimpleController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        logger.info(f"Entering control loop of {self.__class__.__name__}")
        engine = fl_ctx.get_engine()

        # Wait till receiver is done. Otherwise, the job ends.
        receiver = engine.get_component("receiver")
        while not receiver.is_done():
            time.sleep(0.2)

        logger.info("Control flow ends")

    def start_controller(self, fl_ctx: FLContext):
        logger.info("Start controller")

    def stop_controller(self, fl_ctx: FLContext):
        logger.info("Stop controller")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        raise RuntimeError(f"Unknown task: {task_name} from client {client.name}.")
