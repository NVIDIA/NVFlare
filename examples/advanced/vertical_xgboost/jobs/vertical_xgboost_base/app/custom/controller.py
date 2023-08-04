# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
import os

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller, Task
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from trainer import SupportedTasks

import xgboost.federated


class XGBoostController(Controller):
    def __init__(self, port: int, world_size: int, server_key_path: str = None,
                 server_cert_path: str = None, client_cert_path: str = None):
        """Controller for federated XGBoost.

        Args:
            port: the port for the gRPC server to listen on.
            world_size: the number of sites.
            server_key_path: the path to the server key file.
            server_cert_path: the path to the server certificate file.
            client_cert_path: the path to the client certificate file.
        """
        super().__init__()
        self._port = port
        self._world_size = world_size
        self._server_key_path = server_key_path
        self._server_cert_path = server_cert_path
        self._client_cert_path = client_cert_path
        self._server = None

    def start_controller(self, fl_ctx: FLContext):
        self._server = multiprocessing.Process(
            target=xgboost.federated.run_federated_server,
            args=(self._port, self._world_size, self._server_key_path,
                  self._server_cert_path, self._client_cert_path))
        self._server.start()

    def stop_controller(self, fl_ctx: FLContext):
        if self._server:
            self._server.terminate()

    def process_result_of_unknown_task(self, client: Client, task_name: str,
                                       client_task_id: str, result: Shareable,
                                       fl_ctx: FLContext):
        self.log_warning(fl_ctx, f"Unknown task: {task_name} from client {client.name}.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, "XGBoost training control flow started.")
        if abort_signal.triggered:
            return
        task = Task(name=SupportedTasks.TRAIN, data=Shareable())
        self.broadcast_and_wait(
            task=task,
            min_responses=self._world_size,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=1,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return

        self.log_info(fl_ctx, "XGBoost training control flow finished.")
