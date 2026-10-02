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

"""Exercise rank command ordering and launcher shutdown without training dependencies."""

import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.executors import multi_process_executor as mpe
from nvflare.fuel.common.multi_process_executor_constants import MultiProcessCommandNames
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.defs import CellChannel
from nvflare.private.event import fire_event
from nvflare.private.fed.app.client.sub_worker_process import EventRelayer, SubWorkerExecutor

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="requires Unix process groups",
)

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "multi_process_shutdown_worker.py"


class _SentinelCell:
    def __init__(self, directory):
        self.directory = directory

    def fire_and_forget(self, **kwargs):
        assert kwargs["topic"] == MultiProcessCommandNames.CLOSE
        (self.directory / "close").touch()


class _Executor(mpe.MultiProcessExecutor):
    def get_multi_process_command(self):
        return "unused"


def _executor(worker, directory):
    executor = _Executor()
    executor.targets = ["rank-0", "rank-1"]
    executor.engine = SimpleNamespace(client=SimpleNamespace(cell=_SentinelCell(directory)))
    executor.exe_process = worker
    return executor


@contextmanager
def _start_worker(mode, directory):
    readiness = ["ready-0", "ready-1"] if mode == "launcher" else ["ready"]
    log_path = directory / "worker.log"
    with log_path.open("w") as log:
        worker = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), mode, str(directory)],
            start_new_session=True,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10.0
            while not all((directory / name).exists() for name in readiness):
                if worker.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(f"worker readiness failed: {log_path.read_text()}")
                time.sleep(0.01)
            yield worker
        finally:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5.0)


def _assert_reaped(worker):
    # finalize must have reaped the launcher before the test's cleanup runs.
    with pytest.raises(ChildProcessError):
        os.waitpid(worker.pid, os.WNOHANG)


def test_finalize_allows_rank_exit_and_reaps_launcher(tmp_path):
    with _start_worker("launcher", tmp_path) as worker:
        _executor(worker, tmp_path).finalize(FLContext())

        assert (tmp_path / "close").exists()
        assert (tmp_path / "finished-0").exists()
        assert (tmp_path / "finished-1").exists()
        assert (tmp_path / "launcher-finished").exists()
        assert worker.returncode == 0
        _assert_reaped(worker)


def test_finalize_kills_and_reaps_unresponsive_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(mpe, "_WORKER_SHUTDOWN_TIMEOUT", 0.1)
    monkeypatch.setattr(mpe, "_WORKER_KILL_TIMEOUT", 2.0)
    with _start_worker("hung", tmp_path) as worker:
        start = time.monotonic()
        _executor(worker, tmp_path).finalize(FLContext())
        elapsed = time.monotonic() - start

        assert (tmp_path / "close").exists()
        assert worker.returncode == -signal.SIGKILL
        _assert_reaped(worker)
        assert elapsed < 5.0


def test_end_run_ack_precedes_close_over_live_cells():
    flare_decomposers.register()
    entered = threading.Event()
    release = threading.Event()
    sequence = []

    class EndRunHandler(FLComponent):
        def handle_event(self, event_type, fl_ctx):
            assert event_type == EventType.END_RUN
            entered.set()
            if release.wait(timeout=5.0):
                sequence.append("handled")

    # Cells in one process use Cell's native local transport, including payload
    # serialization and correlated replies, without opening network listeners.
    parent_name = f"server.shutdown-{uuid.uuid4().hex}"
    parent = Cell(parent_name, "tcp://127.0.0.1:8002", secure=False, credentials={})
    rank = Cell(f"{parent_name}.0", "tcp://127.0.0.1:8002", secure=False, credentials={})
    worker = SubWorkerExecutor.__new__(SubWorkerExecutor)
    worker.done = False
    context_manager = FLContextManager()
    relayer = EventRelayer(rank, parent_name, local_rank=0)
    manager = SimpleNamespace(
        new_context=context_manager.new_context,
        get_component=lambda _name: relayer,
        fire_event=lambda event, ctx: fire_event(event, [EndRunHandler()], ctx),
    )
    context_manager.engine = manager
    worker.run_manager = manager
    worker.commands = {
        MultiProcessCommandNames.FIRE_EVENT: worker._handle_event,
        MultiProcessCommandNames.CLOSE: worker._close,
    }

    def receive_command(request):
        is_event = request.get_header(MessageHeaderKey.TOPIC) == MultiProcessCommandNames.FIRE_EVENT
        if is_event:
            assert request.get_header(MessageHeaderKey.REPLY_EXPECTED)
        reply = worker.execute_command(request)
        if is_event:
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            sequence.append("ack")
        else:
            sequence.append("close")
        return reply

    rank.register_request_cb(CellChannel.CLIENT_SUB_WORKER_COMMAND, "*", receive_command)
    executor = _Executor()
    executor.targets = [rank.get_fqcn()]
    executor.engine = SimpleNamespace(client=SimpleNamespace(cell=parent))
    try:
        parent.start()
        rank.start()
        with ThreadPoolExecutor(max_workers=1) as pool:
            shutdown = pool.submit(executor.handle_event, EventType.END_RUN, FLContext())
            try:
                assert entered.wait(timeout=5.0)
                assert not shutdown.done()
                assert not worker.done
            finally:
                release.set()
            shutdown.result(timeout=5.0)
        assert worker.done
        assert sequence == ["handled", "ack", "close"]
        assert not executor._abort_requested
    finally:
        release.set()
        for cell in (rank, parent):
            cell.stop()
            CoreCell.ALL_CELLS.pop(cell.get_fqcn(), None)


def test_finalize_kills_remaining_rank_after_launcher_exits(tmp_path):
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), "exited-launcher", str(tmp_path)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        try:
            assert worker.wait(timeout=10.0) == 0
            assert (tmp_path / "ready-0").exists()
            # The live rank inherited the launcher's stdout, keeping the pipe
            # open after the launcher exited. EOF proves that rank also exited.
            assert not select.select([worker.stdout], [], [], 0)[0]
            executor = _executor(worker, tmp_path)
            executor.engine.client.cell = SimpleNamespace(
                fire_and_forget=lambda **_kwargs: {"rank-0": ReturnCode.COMM_ERROR}
            )

            executor.finalize(FLContext())

            assert select.select([worker.stdout], [], [], 2.0)[0]
            assert os.read(worker.stdout.fileno(), 1) == b""
            assert not (tmp_path / "close").exists()
        finally:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5.0)
            worker.stdout.close()
