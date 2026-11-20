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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_opt.flower.connectors.flower_connector import FlowerClientConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.utils import FLEET_PATH_PREFIX, request_to_shareable, shareable_to_response
from nvflare.fuel.utils.network_utils import get_local_addresses
from nvflare.security.logging import secure_format_exception

_LOCALHOST = "127.0.0.1"


class _ProxyHandler(BaseHTTPRequestHandler):
    """Local HTTP proxy that the Flower SuperNode talks to instead of the SuperLink."""

    protocol_version = "HTTP/1.1"  # allow the SuperNode's HTTP client to keep connections alive
    connector: "HttpClientConnector" = None  # set on the per-server subclass

    def log_message(self, format, *args):
        # route access logs to the connector's logger instead of stderr
        self.connector.logger.debug(f"proxy: {format % args}")

    def _reply(self, status: int, headers: Optional[dict] = None, body: bytes = b"", close: bool = False):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_body(self) -> Optional[bytes]:
        """Read the request body. Returns None (after replying with an error) if the body cannot be read."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self._reply(411, close=True)
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("negative length")
        except ValueError:
            self._reply(400, close=True)
            return None
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        if not self.path.startswith(FLEET_PATH_PREFIX):
            # refuse, and close the connection since the unread body would corrupt the next request on it
            self._reply(404, close=True)
            return

        body = self._read_body()
        if body is None:
            return

        status, headers, data = self.connector.forward(self.path, self.headers, body)
        self._reply(status, headers, data)

    def _not_found(self):
        self._reply(404, close=True)

    do_GET = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _not_found


class HttpClientConnector(FlowerClientConnector):
    def __init__(
        self,
        per_msg_timeout=2.0,
        tx_timeout=10.0,
        client_shutdown_timeout=0.5,
    ):
        """Constructor of HttpClientConnector.

        HttpClientConnector connects the FLARE client with the Flower SuperNode. It runs a local HTTP proxy that the
        SuperNode uses as its "SuperLink". Every Fleet API request is sent to the FLARE server with ReliableMessage,
        replayed against the real SuperLink there, and the response is returned to the SuperNode.

        Args:
            per_msg_timeout: per-message timeout for using ReliableMessage
            tx_timeout: transaction timeout for using ReliableMessage
            client_shutdown_timeout: max time for shutting down Flare client
        """
        FlowerClientConnector.__init__(self, per_msg_timeout, tx_timeout)
        self.client_shutdown_timeout = client_shutdown_timeout
        self.proxy_server: Optional[ThreadingHTTPServer] = None
        self.proxy_thread: Optional[threading.Thread] = None
        self.stopped = False
        self.proxy_addr = None
        self._training_stopped = False
        self._client_name = None
        self._stopping = False

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self._client_name = fl_ctx.get_identity_name()

    def _start_client(self, superlink_addr: str, clientapp_api_addr: str, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_CLIENT_NAME: self._client_name,
            Constant.APP_CTX_SUPERLINK_ADDR: superlink_addr,
            Constant.APP_CTX_CLIENTAPP_API_ADDR: clientapp_api_addr,
            Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_client(self):
        self._training_stopped = True

        # The Fleet API has no way to tell a SuperNode to exit, and the SuperNode retries forever, so terminate it.
        self.logger.debug("about to stop applet")
        self.stop_applet(self.client_shutdown_timeout)

    def _is_stopped(self) -> (bool, int):
        applet_stopped, ec = self.is_applet_stopped()
        if applet_stopped:
            return applet_stopped, ec

        if self._training_stopped:
            return True, 0

        if self._stopping:
            self.stop(fl_ctx=None)
            return True, 0

        return False, 0

    def start(self, fl_ctx: FLContext):
        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        # get addresses for flower supernode:
        # - the local proxy address that the supernode connects to (as its superlink)
        # - clientapp_api_addr for client app to connect to the supernode
        addresses = get_local_addresses(2)
        proxy_addr = addresses[0]
        clientapp_api_addr = addresses[1]
        self._start_proxy(proxy_addr)
        self.logger.info(f"Started internal HTTP proxy at {proxy_addr}")
        self._start_client(proxy_addr, clientapp_api_addr, fl_ctx)
        self.logger.info("Started external Flower SuperNode")

    def _start_proxy(self, proxy_addr: str):
        """Start the local HTTP proxy on proxy_addr ("host:port") in a background thread."""
        port = proxy_addr.rpartition(":")[2]
        handler = type("ProxyHandler", (_ProxyHandler,), {"connector": self})
        server = ThreadingHTTPServer((_LOCALHOST, int(port)), handler)
        server.daemon_threads = True
        self.proxy_server = server
        self.proxy_addr = proxy_addr
        self.proxy_thread = threading.Thread(target=server.serve_forever, name="flower-proxy", daemon=True)
        self.proxy_thread.start()

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        self._stop_client()

        server = self.proxy_server
        self.proxy_server = None
        if server:
            self.logger.info("Stop internal HTTP proxy")
            server.shutdown()
            server.server_close()

    def _abort(self, reason: str):
        # stop the proxy (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def forward(self, path: str, headers, body: bytes):
        """Forward a request received from the Flower SuperNode to the FLARE server and return the HTTP reply.

        1. pack the request into a Shareable
        2. send it to the FLARE server with ReliableMessage
        3. unpack the Shareable reply to (status, headers, body)

        If the FLARE side has stopped or the server returns a non-OK return code, the SuperNode gets HTTP 503 and the
        connector starts stopping. The SuperNode itself is terminated by the applet: it never gives up by itself.

        Args:
            path: HTTP path of the request
            headers: HTTP headers of the request
            body: HTTP body of the request

        Returns: tuple of (status, headers, body)
        """
        try:
            if self.stopped:
                self._stopping = True
                self.logger.debug("stopped: replying 503 to supernode (1)")
                return 503, {}, b""

            reply = self._send_flower_request(request_to_shareable(path, headers, body))
            rc = reply.get_return_code()
            if rc == ReturnCode.OK:
                return shareable_to_response(reply)
            else:
                # server side already ended
                self.logger.warning(f"Flower server has stopped with RC {rc}")
                self._stopping = True
                self.logger.debug("stopped: replying 503 to supernode (2)")
                return 503, {}, b""
        except Exception as ex:
            self._abort(reason=f"_send_flower_request exception: {secure_format_exception(ex)}")
            self._stopping = True
            self.logger.debug("stopped: replying 503 to supernode (3)")
            return 503, {}, b""
