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

import threading

import xgboost.federated as xgb_federated

from nvflare.apis.signal import Signal
from nvflare.app_common.xgb.adaptors.grpc.server_adaptor import GrpcServerAdaptor
from nvflare.app_common.xgb.defs import Constant
from nvflare.security.logging import secure_format_exception


class InProcessGrpcServerAdaptor(GrpcServerAdaptor):
    def __init__(self, xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT):
        GrpcServerAdaptor.__init__(self, xgb_server_ready_timeout)
        self.server = None
        self.server_stopped = False

    def _try_start_server(self, port: int, world_size: int, abort_signal: Signal):
        try:
            xgb_federated.run_federated_server(
                port=port,
                world_size=world_size,
            )
        except Exception as ex:
            self.logger.error(f"Exception xgb_federated.run_federated_server: {secure_format_exception(ex)}")
            abort_signal.trigger(True)

    def start_server(self, addr: str, port: int, world_size: int):
        # create a thread to run the server locally.
        # use abort_signal to communicate error back to this thread
        abort_signal = Signal()
        t = threading.Thread(target=self._try_start_server, args=(port, world_size, abort_signal), daemon=True)
        t.start()
        if abort_signal.triggered:
            raise RuntimeError("cannot start XGB server")

    def stop_server(self):
        # currently there is no way to stop XGB server once started
        self.server_stopped = True

    def is_server_stopped(self) -> (bool, int):
        return self.server_stopped, 0
