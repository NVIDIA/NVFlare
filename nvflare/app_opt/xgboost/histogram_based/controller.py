# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.security.logging import secure_format_exception, secure_format_traceback

from .constants import XGB_TRAIN_TASK, XGBShareableHeader


class XGBFedController(Controller):
    def __init__(self, train_timeout: int = 300, port: int = None):
        """Federated XGBoost training controller for histogram-base collaboration.

        It starts the XGBoost federated server and kicks off all the XGBoost job on
        each NVFlare client. The configuration is generic for this component and
        no modification is needed for most training jobs.

        Args:
            train_timeout (int, optional): Time to wait for clients to do local training in seconds.
            port (int, optional): the port to open XGBoost FL server

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__()

        if not isinstance(train_timeout, int):
            raise TypeError("train_timeout must be int but got {}".format(type(train_timeout)))

        self._port = port
        self._xgb_fl_server = None
        self._participate_clients = None
        self._rank_map = None
        self._secure = False
        self._train_timeout = train_timeout
        self._server_cert_path = None
        self._server_key_path = None
        self._ca_cert_path = None
        self._started = False

    def _get_certificates(self, fl_ctx: FLContext):
        workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        bin_folder = workspace.get_startup_kit_dir()
        server_cert_path = os.path.join(bin_folder, "server.crt")
        if not os.path.exists(server_cert_path):
            self.log_error(fl_ctx, "Missing server certificate (server.crt)")
            return False
        server_key_path = os.path.join(bin_folder, "server.key")
        if not os.path.exists(server_key_path):
            self.log_error(fl_ctx, "Missing server key (server.key)")
            return False
        ca_cert_path = os.path.join(bin_folder, "rootCA.pem")
        if not os.path.exists(ca_cert_path):
            self.log_error(fl_ctx, "Missing ca certificate (rootCA.pem)")
            return False
        self._server_cert_path = server_cert_path
        self._server_key_path = server_key_path
        self._ca_cert_path = ca_cert_path
        return True

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Initializing {self.__class__.__name__} workflow.")
        xgb_federated, flag = optional_import(module="xgboost.federated")
        if not flag:
            self.log_error(fl_ctx, "Can't import xgboost.federated")
            return

        # Assumption: all clients are used
        clients = self._engine.get_clients()
        # Sort by client name so rank is consistent
        clients.sort(key=lambda client: client.name)
        rank_map = {clients[i].name: i for i in range(0, len(clients))}
        self._rank_map = rank_map
        self._participate_clients = clients

        if not self._port:
            self._port = get_open_ports(1)[0]

        self.log_info(fl_ctx, f"Starting XGBoost FL server on port {self._port}")

        self._secure = self._engine.server.secure_train
        if self._secure:
            if not self._get_certificates(fl_ctx):
                self.log_error(fl_ctx, "Can't get required certificates for XGB FL server in secure mode.")
                return
            self._xgb_fl_server = multiprocessing.Process(
                target=xgb_federated.run_federated_server,
                args=(self._port, len(clients), self._server_key_path, self._server_cert_path, self._ca_cert_path),
            )
        else:
            self._xgb_fl_server = multiprocessing.Process(
                target=xgb_federated.run_federated_server, args=(self._port, len(clients))
            )
        self._xgb_fl_server.start()
        self._started = True

    def stop_controller(self, fl_ctx: FLContext):
        if self._xgb_fl_server:
            self._xgb_fl_server.terminate()
        self._started = False

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ):
        self.log_error(fl_ctx, f"Unknown task: {task_name} from client {client.name}.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Begin XGBoost training phase.")
        if not self._started:
            msg = "Controller does not start successfully."
            self.log_error(fl_ctx, msg)
            self.system_panic(msg, fl_ctx)
            return

        try:
            data = Shareable()
            data.set_header(XGBShareableHeader.WORLD_SIZE, len(self._participate_clients))
            data.set_header(XGBShareableHeader.RANK_MAP, self._rank_map)
            data.set_header(XGBShareableHeader.XGB_FL_SERVER_PORT, self._port)
            data.set_header(XGBShareableHeader.XGB_FL_SERVER_SECURE, self._secure)

            train_task = Task(
                name=XGB_TRAIN_TASK,
                data=data,
                timeout=self._train_timeout,
            )

            self.broadcast_and_wait(
                task=train_task,
                targets=self._participate_clients,
                min_responses=len(self._participate_clients),
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            self.log_info(fl_ctx, "Finish training phase.")

        except Exception as e:
            err = secure_format_traceback()
            error_msg = f"Exception in control_flow: {secure_format_exception(e)}: {err}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(secure_format_exception(e), fl_ctx)
