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

"""Real socket and process lifecycle contracts; no Flower installation needed."""

import http.client
import os
import shlex
import signal
import socket
import sys
import threading
import time
from unittest.mock import MagicMock

import psutil
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tie.process_mgr import CommandDescriptor
from nvflare.app_opt.flower.http_transport import FleetHttpClient, FleetHttpServer, HttpError, http_response
from nvflare.app_opt.flower.process import start_process
from tests.timing_utils import ThreadGroup, wait_for


def envelope(path, body=b""):
    value = Shareable()
    value.update(
        method="POST",
        path=path,
        body=body,
        headers={"content-type": "application/protobuf", "flwr-signature": "opaque-signature"},
    )
    return value


@pytest.fixture
def server_factory():
    servers = []

    def create(callback, **kwargs):
        server = FleetHttpServer(callback, **kwargs)
        servers.append(server)
        server.start()
        return server

    yield create
    for server in reversed(servers):
        server.close()


def test_binary_roundtrip_and_error_semantics(server_factory):
    seen = []

    def forward(req):
        seen.append(req)
        return http_response(429, req["body"], {"retry-after": "7", "content-type": "application/protobuf"})

    server = server_factory(forward)
    client = FleetHttpClient(server.address)
    try:
        payload = bytes(range(256)) * (32 * 1024)  # 8 MiB, above the old default gRPC limit
        result = client.send(envelope("/v1/fleet/push-object", payload))
        assert result["status"] == 429
        assert result["body"] == payload
        assert result["headers"]["retry-after"] == "7"
        assert len(seen) == 1  # an HTTP error must not cause a hidden retry
        assert seen[0]["headers"]["flwr-signature"] == "opaque-signature"
    finally:
        client.close()


@pytest.mark.parametrize(
    "path,method,body,headers,status",
    [
        ("/v1/control/start-run", "POST", b"", {}, 404),
        ("/v1/fleet/get-run", "GET", b"", {}, 405),
        ("/v1/fleet/get-run", "POST", b"x" * 33, {}, 413),
        ("/v1/fleet/get-run", "POST", b"", {"Transfer-Encoding": "chunked", "Content-Length": "0"}, 400),
        ("/v1/fleet/get-run", "POST", b"", {"Content-Type": "application/json"}, 415),
    ],
)
def test_invalid_local_requests_never_forward(server_factory, path, method, body, headers, status):
    seen = []
    server = server_factory(lambda req: seen.append(req), max_message_bytes=32)
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        assert response.status == status
        response.read()
        assert not seen
    finally:
        conn.close()


def test_heartbeat_progresses_while_bulk_workers_are_blocked(server_factory):
    entered = [threading.Event() for _ in range(4)]
    release = threading.Event()
    lock = threading.Lock()
    counter = 0

    def forward(req):
        nonlocal counter
        if req["path"] == "/v1/fleet/pull-object":
            with lock:
                index = counter
                counter += 1
            entered[index].set()
            release.wait()
        return http_response(200, b"ok")

    server = server_factory(forward)
    client = FleetHttpClient(server.address)
    try:
        with ThreadGroup(release) as workers:
            for _ in entered:
                workers.start(client.send, envelope("/v1/fleet/pull-object"))
            assert all(event.wait(10) for event in entered)
            assert client.send(envelope("/v1/fleet/pull-object"))["status"] == 503
            assert client.send(envelope("/v1/fleet/send-node-heartbeat"))["body"] == b"ok"
    finally:
        release.set()
        client.close()


def test_close_unblocks_partial_http_body():
    server = FleetHttpServer(lambda req: http_response(200))
    server.start()
    conn = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
    try:
        conn.sendall(b"POST /v1/fleet/get-run HTTP/1.1\r\nContent-Length: 1000\r\n\r\n")
        wait_for(lambda: bool(server._sockets))
        start = time.monotonic()
        with ThreadGroup() as workers:
            worker = workers.start(server.close)
            worker.join(5)
            assert not worker.is_alive()
        assert time.monotonic() - start < 5
        assert not server._sockets
    finally:
        conn.close()
        server.close()


@pytest.mark.skipif(os.name != "posix", reason="Flower integration requires POSIX process groups")
def test_stops_owned_detached_descendant(tmp_path):
    ready = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        + f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        + f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); time.sleep(300)"
    )
    ctx = MagicMock(spec=FLContext)
    ctx.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    manager = None
    child = None
    try:
        manager = start_process(
            CommandDescriptor(
                shlex.join([sys.executable, "-c", parent_code]), log_stdout=False, log_file_name="process.log"
            ),
            ctx,
        )
        wait_for(ready.exists)
        child = psutil.Process(int(ready.read_text()))
        assert child.is_running()
        manager.stop(timeout=0.2)
        assert manager.process.poll() is not None
        wait_for(lambda: not child.is_running() or child.status() == psutil.STATUS_ZOMBIE)
        assert not manager._log_writer.is_alive()
    finally:
        descendants = [child] if child else []
        if manager:
            # Discover even a detached child that has not written its readiness
            # file yet. Cleanup cannot depend on readiness or stop() succeeding.
            try:
                descendants.extend(psutil.Process(manager.process.pid).children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            for owned in reversed(descendants):
                try:
                    owned.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(manager.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            manager.process.wait(timeout=10)
            manager._log_writer.join(timeout=10)
            assert not manager._log_writer.is_alive()
            manager.process.stdout.close()
            if manager.log_file:
                manager.log_file.close()


def test_truncated_protobuf_is_not_reframed_as_success():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(10)
    client = FleetHttpClient(f"127.0.0.1:{listener.getsockname()[1]}")

    def upstream():
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(10)
            headers = b""
            while b"\r\n\r\n" not in headers:
                chunk = conn.recv(4096)
                assert chunk
                headers += chunk
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n\x08\x01")

    try:
        with ThreadGroup() as workers:
            workers.start(upstream)
            with pytest.raises(HttpError, match="Incomplete SuperLink response") as error:
                client.send(envelope("/v1/fleet/get-run"))
            assert error.value.status == 502
    finally:
        client.close()
        listener.close()
