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

"""Forward authenticated NVFlare site requests to the job's local SuperLink."""

import http.client
import threading

from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.app_opt.flower.connectors.flower_connector import FlowerServerConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.http_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    FleetHttpClient,
    HttpError,
    error_response,
    http_response,
    validate_request,
)
from nvflare.fuel.utils.network_utils import get_local_addresses


class HttpServerConnector(FlowerServerConnector):
    def __init__(self, monitor_interval=0.5, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
        super().__init__(monitor_interval)
        self.max_message_bytes = max_message_bytes
        self.http_client = None
        self._stopped = False
        self._identity_lock = threading.Lock()
        self._key_owners = {}

    def start(self, fl_ctx):
        address = get_local_addresses(1)[0]
        self.http_client = FleetHttpClient(address, self.max_message_bytes)
        try:
            self.start_applet(
                {Constant.APP_CTX_SUPERLINK_ADDR: address, Constant.APP_CTX_NUM_ROUNDS: self.num_rounds}, fl_ctx
            )
        except Exception:
            self.stop(fl_ctx)
            raise

    def stop(self, fl_ctx):
        self._stopped = True
        if self.http_client:
            self.http_client.close()
        self.stop_applet()

    def process_app_request(self, op, request, fl_ctx, abort_signal):
        # Status polling runs in the monitor. Never serialize heartbeats behind
        # a slow `flwr list` invocation on the request path.
        if self._stopped or abort_signal.triggered:
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
        if op != "request":
            return http_response(400)
        return self.send_request_to_flower(request, fl_ctx)

    def send_request_to_flower(self, request, fl_ctx):
        try:
            path, headers, _ = validate_request(request, self.max_message_bytes)
            peer = fl_ctx.get_peer_context()
            site = peer.get_identity_name() if peer else None
            key = headers.get("flwr-public-key")
            if not site or not key or not headers.get("flwr-signature") or not headers.get("flwr-timestamp"):
                return http_response(403)
            # A Flower key may only be used by its registering NVFlare site in
            # this job. SuperLink verifies signatures and node/body identity.
            with self._identity_lock:
                owner = self._key_owners.get(key)
                if owner and owner != site:
                    return http_response(403)
                if not owner:
                    if path != "/v1/fleet/register-node":
                        return http_response(403)
                    self._key_owners[key] = site
            try:
                reply = self.http_client.send(request)
            except Exception:
                # Do not release an identity after an uncertain registration.
                # The job will fail rather than risk a cross-site takeover.
                raise
            if path == "/v1/fleet/register-node" and reply["status"] >= 400 and not owner:
                with self._identity_lock:
                    self._key_owners.pop(key, None)
            return reply
        except HttpError as ex:
            return error_response(ex)
        except TimeoutError:
            return http_response(504)
        except (OSError, http.client.HTTPException):
            return http_response(502)
