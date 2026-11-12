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

"""Bounded, loopback-only transport for Flower's unary protobuf Fleet API.

Only the envelope is interpreted. Model objects, messages and Flower errors remain
opaque bytes. ReliableMessage owns NVFlare retries; this layer never retries an
HTTP operation or follows redirects.
"""

import http.client
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.validation_utils import check_positive_int

DEFAULT_MAX_MESSAGE_BYTES = 512 * 1024 * 1024
FLEET_PATHS = frozenset(
    f"/v1/fleet/{name}"
    for name in (
        "register-node",
        "activate-node",
        "deactivate-node",
        "unregister-node",
        "send-node-heartbeat",
        "pull-messages",
        "push-messages",
        "get-run",
        "get-fab",
        "push-object",
        "pull-object",
        "confirm-message-received",
    )
)
HEARTBEAT_PATH = "/v1/fleet/send-node-heartbeat"
BULK_PATHS = frozenset(f"/v1/fleet/{name}" for name in ("push-object", "pull-object", "get-fab"))
REQUEST_HEADERS = frozenset(
    (
        "content-type",
        "accept",
        "flwr-public-key",
        "flwr-signature",
        "flwr-timestamp",
        "flwr-package-name",
        "flwr-package-version",
        "flwr-component-name",
    )
)
RESPONSE_HEADERS = REQUEST_HEADERS | {"retry-after", "www-authenticate"}


class HttpError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def filter_headers(headers, allowed):
    result = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise HttpError(400, "Invalid HTTP header")
        key = key.lower()
        if key not in allowed:
            continue
        if key in result or len(value) > 8192 or any(c in value for c in "\r\n\x00"):
            raise HttpError(400, "Invalid HTTP header")
        result[key] = value
    return result


def validate_request(request, max_message_bytes):
    if request.get("method") != "POST":
        raise HttpError(405, "Only POST is supported")
    if request.get("path") not in FLEET_PATHS:
        raise HttpError(404, "Unknown Fleet route")
    headers = request.get("headers")
    if not isinstance(headers, dict):
        raise HttpError(400, "Invalid HTTP headers")
    headers = filter_headers(headers, REQUEST_HEADERS)
    if headers.get("content-type") != "application/protobuf":
        raise HttpError(415, "Expected application/protobuf")
    body = request.get("body")
    if not isinstance(body, bytes):
        raise HttpError(400, "Expected a binary protobuf body")
    if len(body) > max_message_bytes:
        raise HttpError(413, "Fleet payload exceeds max_message_bytes")
    return request["path"], headers, body


def http_response(status, body=b"", headers=None):
    response = Shareable()
    response.update(status=status, headers=headers or {}, body=body)
    return response


def error_response(error):
    return http_response(error.status, str(error).encode(), {"content-type": "text/plain"})


def split_address(address):
    host, port = address.rsplit(":", 1)
    if host != "127.0.0.1" or not 0 < int(port) < 65536:
        raise ValueError("Flower HTTP endpoints must use 127.0.0.1 and a valid port")
    return host, int(port)


class FleetHttpClient:
    """Forward one validated request to a fixed, job-local SuperLink."""

    def __init__(self, address, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
        self.host, self.port = split_address(address)
        check_positive_int("max_message_bytes", max_message_bytes)
        self.max_message_bytes = max_message_bytes
        self._lock = threading.Lock()
        self._connections = set()
        self._closed = False

    def send(self, request):
        path, headers, body = validate_request(request, self.max_message_bytes)
        timeout = 3.0 if path == HEARTBEAT_PATH else 20.0
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        with self._lock:
            if self._closed:
                return http_response(503)
            # Connect under the lock so close() cannot miss an in-flight socket.
            conn.connect()
            self._connections.add(conn)
        try:
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    length = int(length)
                except ValueError as ex:
                    raise HttpError(502, "Invalid SuperLink Content-Length") from ex
                if length < 0 or length > self.max_message_bytes:
                    raise HttpError(502, "Invalid or oversized SuperLink response")
            content = response.read(self.max_message_bytes + 1)
            if len(content) > self.max_message_bytes:
                raise HttpError(502, "SuperLink response exceeds max_message_bytes")
            # HTTPResponse.read(amt) can silently return fewer bytes at EOF.
            # Do not reframe a truncated protobuf as a successful shorter body.
            if length is not None and len(content) != length:
                raise HttpError(502, "Incomplete SuperLink response")
            return http_response(response.status, content, filter_headers(response.headers, RESPONSE_HEADERS))
        finally:
            with self._lock:
                self._connections.discard(conn)
            conn.close()

    def close(self):
        with self._lock:
            self._closed = True
            for conn in self._connections:
                if conn.sock:
                    _shutdown_socket(conn.sock)


def _shutdown_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class _FleetHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        # Requests contain signed identity headers and model data: do not log them.
        pass

    def do_POST(self):
        gate = self.server.bulk_slots if self.path in BULK_PATHS else self.server.control_slots
        if not gate.acquire(blocking=False):
            self._reply(http_response(503, headers={"retry-after": "1"}))
            return
        try:
            if self.path not in FLEET_PATHS:
                raise HttpError(404, "Unknown Fleet route")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                raise HttpError(411, "A single Content-Length is required")
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                raise HttpError(400, "Encoded request bodies are not supported")
            length = int(lengths[0])
            if length > self.server.max_message_bytes:
                raise HttpError(413, "Fleet payload exceeds max_message_bytes")
            headers = filter_headers(self.headers, REQUEST_HEADERS)
            body = self.rfile.read(length)
            if len(body) != length:
                raise HttpError(400, "Incomplete request body")
            request = Shareable()
            request.update(method="POST", path=self.path, headers=headers, body=body)
            validate_request(request, self.server.max_message_bytes)
            reply = self.server.forward(request)
            status, content = reply.get("status"), reply.get("body")
            if type(status) is not int or not 200 <= status <= 599 or not isinstance(content, bytes):
                raise HttpError(502, "Invalid Fleet response")
            if len(content) > self.server.max_message_bytes:
                raise HttpError(502, "Fleet response exceeds max_message_bytes")
            reply["headers"] = filter_headers(reply["headers"], RESPONSE_HEADERS)
        except HttpError as ex:
            reply = error_response(ex)
        except (TimeoutError, socket.timeout):
            reply = http_response(504)
        except Exception:
            reply = http_response(502)
        finally:
            gate.release()
        self._reply(reply)

    def _reply(self, reply):
        self.close_connection = True
        try:
            self.send_response(reply["status"])
            for key, value in reply["headers"].items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(reply["body"])))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(reply["body"])
        except OSError:
            pass  # The local SuperNode may already have timed out or stopped.

    def do_GET(self):
        self._reply(http_response(405))

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_GET


class FleetHttpServer(ThreadingHTTPServer):
    """Local Fleet endpoint with bounded workers and reserved non-bulk capacity.

    The owner must cancel its forwarding callback before close(). Non-daemon
    workers are joined, including clients that connected without sending a body.
    """

    daemon_threads = False
    allow_reuse_address = False

    def __init__(self, forward, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
        check_positive_int("max_message_bytes", max_message_bytes)
        self.forward = forward
        self.max_message_bytes = max_message_bytes
        self.bulk_slots = threading.BoundedSemaphore(4)
        self.control_slots = threading.BoundedSemaphore(16)
        self._slots = threading.BoundedSemaphore(32)
        self._lock = threading.Lock()
        self._sockets = set()
        super().__init__(("127.0.0.1", 0), _FleetHandler)
        self.address = f"127.0.0.1:{self.server_port}"
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1})

    def start(self):
        self._thread.start()

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        request.settimeout(30.0)
        with self._lock:
            self._sockets.add(request)
        try:
            super().process_request(request, client_address)
        except Exception:
            self._release(request)
            raise

    def _release(self, request):
        with self._lock:
            self._sockets.discard(request)
        self._slots.release()

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(request)

    def close(self):
        self.shutdown()
        self._thread.join()
        with self._lock:
            for sock in self._sockets:
                _shutdown_socket(sock)
        self.server_close()
