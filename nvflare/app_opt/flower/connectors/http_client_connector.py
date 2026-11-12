# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Bridge a local Flower SuperNode to NVFlare ReliableMessage."""

from nvflare.apis.shareable import ReturnCode
from nvflare.apis.signal import Signal
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_opt.flower.connectors.flower_connector import FlowerClientConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.http_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    HEARTBEAT_PATH,
    FleetHttpServer,
    http_response,
)
from nvflare.fuel.utils.network_utils import get_local_addresses


class HttpClientConnector(FlowerClientConnector):
    def __init__(
        self,
        per_msg_timeout=2.0,
        tx_timeout=25.0,
        client_shutdown_timeout=5.0,
        max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES,
    ):
        super().__init__(per_msg_timeout, tx_timeout)
        self.client_shutdown_timeout = client_shutdown_timeout
        self.max_message_bytes = max_message_bytes
        self.http_server = None
        self._server_done = False
        self._exit_code = 0

    def set_abort_signal(self, abort_signal):
        # Local shutdown cancels forwarding without aborting the enclosing job.
        super().set_abort_signal(Signal(parent=abort_signal))

    def start(self, fl_ctx):
        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")
        self.http_server = FleetHttpServer(self._forward, self.max_message_bytes)
        self.http_server.start()
        try:
            self.start_applet(
                {
                    Constant.APP_CTX_SUPERLINK_ADDR: self.http_server.address,
                    Constant.APP_CTX_CLIENTAPP_API_ADDR: get_local_addresses(1)[0],
                    Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
                },
                fl_ctx,
            )
        except Exception:
            self.stop(fl_ctx)
            raise

    def _forward(self, request):
        if self.stopped or self._server_done or self.abort_signal.triggered:
            return http_response(503)
        heartbeat = request["path"] == HEARTBEAT_PATH
        try:
            reply = self.send_request(
                op="request",
                target=None,
                request=request,
                fl_ctx=None,
                per_msg_timeout=min(self.per_msg_timeout, 0.5 if heartbeat else 2.0),
                tx_timeout=min(self.tx_timeout, 3.5 if heartbeat else 25.0),
            )
            rc = reply.get_return_code()
        except Exception:
            self._server_done = True
            self._exit_code = TieConstant.EXIT_CODE_FAILED
            return http_response(503)
        if rc == ReturnCode.OK:
            return reply
        if rc == ReturnCode.SERVICE_UNAVAILABLE:
            self._server_done = True
        elif rc != ReturnCode.TASK_ABORTED:
            # Continuing after an uncertain state-changing request can duplicate
            # Flower operations. Fail the job instead of replaying at this layer.
            self._server_done = True
            self._exit_code = TieConstant.EXIT_CODE_FAILED
        return http_response(503)

    def _is_stopped(self):
        if self._server_done or self.stopped:
            return True, self._exit_code
        return self.is_applet_stopped()

    def stop(self, fl_ctx):
        if self.stopped:
            return
        self.stopped = True
        if self.abort_signal:
            self.abort_signal.trigger(True)
        try:
            self.stop_applet(self.client_shutdown_timeout)
        finally:
            if self.http_server:
                self.http_server.close()
                self.http_server = None
