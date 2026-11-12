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
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import psutil
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tie.process_mgr import CommandDescriptor
from nvflare.app_opt.flower.http_transport import FleetHttpClient, FleetHttpServer, HttpError, http_response
from nvflare.app_opt.flower.process import FlowerProcessManager
from tests.process_test_utils import OwnedProcesses, process_running
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


@pytest.fixture
def flower_process_factory(tmp_path):
    if os.name != "posix":
        pytest.skip("Flower integration requires POSIX process groups")
    ctx = MagicMock(spec=FLContext)
    ctx.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    # A separate marker owns even children orphaned before readiness. Harness
    # cleanup must not depend on the product's stop() or descendant tracking.
    owner = OwnedProcesses(grace=1, process_factory=subprocess.Popen)
    managers = []

    def popen(args, **kwargs):
        kwargs.pop("start_new_session")
        return owner.start(args, **kwargs)

    def create(code):
        manager = FlowerProcessManager(
            CommandDescriptor(shlex.join([sys.executable, "-c", code]), log_stdout=False, log_file_name="process.log")
        )
        managers.append(manager)
        with patch("nvflare.app_opt.flower.process.subprocess.Popen", side_effect=popen):
            manager.start(ctx)
        return manager

    try:
        yield create, managers
    finally:
        errors = []

        def attempt(action):
            try:
                action()
            except Exception as error:
                errors.append(error)

        attempt(owner.close)
        for manager in managers:
            manager._monitor_stop.set()
            manager._log_stop.set()
        for manager in managers:
            for thread in (manager._monitor, manager._log_writer):
                if thread and thread.ident is not None:
                    attempt(lambda thread=thread: thread.join(timeout=10))
                    if thread.is_alive():
                        errors.append(AssertionError("test-owned Flower thread survived cleanup"))
            if manager.process:
                attempt(lambda: manager.process.wait(timeout=10))
                attempt(manager.process.stdout.close)
            if manager.log_file:
                attempt(manager.log_file.close)
        if errors:
            raise RuntimeError(f"Flower test cleanup failed: {errors}") from errors[0]


@pytest.fixture
def detached_flower_factory(flower_process_factory, tmp_path):
    factory, _ = flower_process_factory

    def create(*, clear_marker=False, count=1):
        parent_code = "import os,subprocess,sys,time; env=os.environ.copy(); "
        if clear_marker:
            parent_code += "env.pop('NVFLARE_FLOWER_PROCESS_OWNER',None); "
        ready_dir = tmp_path / f"children-{time.monotonic_ns()}"
        ready_dir.mkdir()
        ready_files = [ready_dir / f"child-{i}.pid" for i in range(count)]
        for ready in ready_files:
            child_code = (
                "import os,signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "sys.stdout.write('detached log tail'); sys.stdout.flush(); "
                + f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)"
            )
            parent_code += f"subprocess.Popen([sys.executable,'-c',{child_code!r}],env=env,start_new_session=True); "
        manager = factory(parent_code + "time.sleep(300)")
        children = []
        for ready in ready_files:
            wait_for(lambda ready=ready: ready.exists() and ready.read_text().isdigit())
            children.append(psutil.Process(int(ready.read_text())))
        return manager, children

    return create


@pytest.mark.parametrize("scenario", ["running", "parent-exited", "unobserved-parent-exit"])
def test_stops_owned_detached_descendant(detached_flower_factory, monkeypatch, scenario):
    if scenario == "unobserved-parent-exit":
        # Force the window where a worker is orphaned before any tree snapshot.
        monkeypatch.setattr(FlowerProcessManager, "_monitor_descendants", lambda self: self._monitor_stop.wait())
    manager, (child,) = detached_flower_factory(clear_marker=scenario == "parent-exited")
    assert process_running(child)
    if scenario == "parent-exited":
        # Clearing the marker makes this exercise retained identity ownership.
        wait_for(lambda: child in manager._owned_descendants())
    if scenario != "running":
        manager.process.kill()
        manager.process.wait(timeout=10)
        assert process_running(child)
    manager.stop(timeout=0.2)
    assert manager.process.poll() is not None
    wait_for(lambda: not process_running(child))
    assert not manager._monitor.is_alive()
    assert not manager._log_writer.is_alive()
    assert manager.process.stdout.closed
    assert manager.log_file is None
    assert manager.stop(timeout=0.2) == manager.process.returncode


def test_failed_worker_cleanup_releases_log_reader(detached_flower_factory, tmp_path):
    manager, (blocked, other) = detached_flower_factory(count=2)
    wait_for(lambda: blocked in manager._owned_descendants() and other in manager._owned_descendants())
    manager.process.kill()
    manager.process.wait(timeout=10)
    kill = psutil.Process.kill

    def deny_one_worker(process):
        if process == blocked:
            raise psutil.AccessDenied(process.pid)
        return kill(process)

    with patch.object(psutil.Process, "kill", deny_one_worker):
        with pytest.raises(RuntimeError, match="Flower process cleanup failed"):
            manager.stop(timeout=0.2)
    assert process_running(blocked)  # still holding the log pipe open
    wait_for(lambda: not process_running(other))
    assert not manager._monitor.is_alive()
    assert not manager._log_writer.is_alive()
    assert manager.process.stdout.closed
    assert manager.log_file is None
    assert "detached log tail" in (tmp_path / "process.log").read_text()


@pytest.mark.parametrize("target", ["_monitor_descendants", "_write_log"])
def test_flower_thread_start_failure_cleans_up_process(flower_process_factory, target):
    factory, managers = flower_process_factory
    start = threading.Thread.start

    def fail_log_start(thread):
        if thread._target.__name__ == target:
            raise RuntimeError("intentional thread start failure")
        return start(thread)

    with patch.object(threading.Thread, "start", fail_log_start):
        with pytest.raises(RuntimeError, match="intentional thread start failure"):
            factory("import time; time.sleep(300)")
    manager = managers[0]
    assert manager.process.poll() is not None
    assert not manager._monitor.is_alive()
    assert manager._log_writer is None or not manager._log_writer.is_alive()
    assert manager.process.stdout.closed
    assert manager.log_file is None


def test_flower_cleanup_leaves_another_jobs_orphan_alive(detached_flower_factory):
    first, (first_child,) = detached_flower_factory()
    second, (second_child,) = detached_flower_factory()
    for manager in (first, second):
        manager.process.kill()
        manager.process.wait(timeout=10)
    first.stop(timeout=0.2)
    wait_for(lambda: not process_running(first_child))
    assert process_running(second_child)
    second.stop(timeout=0.2)
    wait_for(lambda: not process_running(second_child))


def test_flower_cleanup_finds_worker_spawned_during_shutdown(flower_process_factory, tmp_path, monkeypatch):
    factory, _ = flower_process_factory
    monkeypatch.setattr(FlowerProcessManager, "_monitor_descendants", lambda self: self._monitor_stop.wait())
    ready, late_ready = tmp_path / "worker.pid", tmp_path / "late.pid"
    late_code = (
        "import os,time; from pathlib import Path; "
        + f"Path({str(late_ready)!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    worker_code = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        + f"signal.signal(signal.SIGUSR1,lambda *_: subprocess.Popen([sys.executable,'-c',{late_code!r}],start_new_session=True)); "
        + f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    manager = factory(
        "import subprocess,sys,time; "
        + f"subprocess.Popen([sys.executable,'-c',{worker_code!r}],start_new_session=True); time.sleep(300)"
    )
    wait_for(lambda: ready.exists() and ready.read_text().isdigit())
    worker = psutil.Process(int(ready.read_text()))
    manager.process.kill()
    manager.process.wait(timeout=10)
    kill = psutil.Process.kill
    spawned = []

    def spawn_before_kill(process):
        if process == worker and not spawned:
            process.send_signal(signal.SIGUSR1)
            wait_for(lambda: late_ready.exists() and late_ready.read_text().isdigit())
            spawned.append(psutil.Process(int(late_ready.read_text())))
        return kill(process)

    with patch.object(psutil.Process, "kill", spawn_before_kill):
        # This functional case launches an interpreter inside the kill hook;
        # its completion budget must include startup under CI contention.
        manager.stop(timeout=10)
    assert spawned
    wait_for(lambda: not process_running(spawned[0]))
    assert not manager._monitor.is_alive()
    assert not manager._log_writer.is_alive()


def test_flower_log_reader_preserves_output(flower_process_factory, tmp_path):
    factory, _ = flower_process_factory
    output = "Flower log: \N{SNOWMAN}\n" * 16384 + "partial final line \N{SNOWMAN}"
    manager = factory("import sys; sys.stdout.write('Flower log: \\u2603\\n' * 16384 + 'partial final line \\u2603')")
    manager.process.wait(timeout=30)
    manager.stop(timeout=0.2)
    assert (tmp_path / "process.log").read_text() == output


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
