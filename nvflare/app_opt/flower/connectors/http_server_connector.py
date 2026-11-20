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
import http.client
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.app_opt.flower.connectors.flower_connector import FlowerServerConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.utils import is_fleet_path, response_to_shareable, shareable_to_request
from nvflare.fuel.utils.network_utils import get_local_addresses
from nvflare.security.logging import secure_format_exception


class HttpServerConnector(FlowerServerConnector):
    def __init__(
        self,
        monitor_interval: float = 1.0,
        request_timeout: float = 300.0,
    ):
        """Constructor of HttpServerConnector.

        HttpServerConnector connects the FLARE server with the Flower SuperLink. Fleet API requests received from
        FLARE clients are replayed as HTTP POST requests against the SuperLink.

        Args:
            monitor_interval: how often to check whether the applet is still running
            request_timeout: timeout in seconds for a single HTTP request to the SuperLink
        """
        FlowerServerConnector.__init__(self, monitor_interval)
        self.request_timeout = request_timeout
        self.superlink_addr: Optional[str] = None
        self._server_stopped = False
        self._exit_code = 0

    def _start_server(self, superlink_api_addr: str, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_SUPERLINK_API_ADDR: superlink_api_addr,
            Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_server(self):
        self._server_stopped = True
        self._exit_code = self.stop_applet()

    def _is_stopped(self) -> (bool, int):
        runner_stopped, ec = self.is_applet_stopped()
        if runner_stopped:
            self.logger.debug("applet is stopped!")
            return runner_stopped, ec

        if self._server_stopped:
            self.logger.info("Flower server connector is stopped!")
            return True, self._exit_code

        return False, 0

    def start(self, fl_ctx: FLContext):
        # The SuperLink serves the Fleet API (for SuperNodes), the Runtime API (for the ServerApp) and the Control API
        # (for "flwr run") on a single HTTP address. We pick a free local address for it.
        try:
            addresses = get_local_addresses(1)
        except Exception as ex:
            raise RuntimeError(f"failed to get address for Flower SuperLink: {secure_format_exception(ex)}")

        self.superlink_addr = addresses[0]
        self.log_info(fl_ctx, f"starting http connector: superlink_addr={self.superlink_addr}")
        self._start_server(self.superlink_addr, fl_ctx)

    def stop(self, fl_ctx: FLContext):
        self._stop_server()

    def send_request_to_flower(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Send the request received from a FL client to the Flower SuperLink.

        The request carries an HTTP path, headers and body. It is replayed as a POST to the SuperLink, and the
        SuperLink's status, headers and body are returned in the reply, errors included.

        Args:
            request: the request received from FL client
            fl_ctx: FL context

        Returns: the SuperLink's HTTP response converted to Shareable
        """
        stopped, _ = self.is_applet_stopped()
        if stopped:
            self.log_warning(fl_ctx, "dropped app request since applet is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        addr = self.superlink_addr
        if not addr:
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        path, headers, body = shareable_to_request(request)
        if not is_fleet_path(path):
            # The SuperLink port also serves the Control and Runtime APIs. Never let a client reach them.
            self.log_error(fl_ctx, f"rejected request with invalid Flower path: {path!r}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        host, _, port = addr.rpartition(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=self.request_timeout)
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            return response_to_shareable(resp.status, resp.getheaders(), data)
        except (OSError, http.client.HTTPException) as ex:
            # The SuperLink is unreachable or died mid-request. Tell the client side the server is gone.
            self.log_warning(fl_ctx, f"exception communicating with Flower SuperLink: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
        finally:
            conn.close()
