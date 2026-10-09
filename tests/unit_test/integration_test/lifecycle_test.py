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

import json
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from tests.integration_test.src import poc_site_launcher, site_launcher, utils
from tests.integration_test.src.poc_site_launcher import POCSiteLauncher
from tests.integration_test.src.site_launcher import ServerProperties
from tests.timing_utils import ManualClock


@pytest.fixture
def clock(monkeypatch):
    clock = ManualClock()
    clock.sleep = clock.advance
    monkeypatch.setattr(utils, "time", clock)
    monkeypatch.setattr(site_launcher, "time", clock)
    return clock


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    launcher = POCSiteLauncher.__new__(POCSiteLauncher)
    launcher.server_properties = {"server": ServerProperties("server", str(tmp_path), Mock(returncode=0), 8003)}
    monkeypatch.setattr(site_launcher, "process_group_alive", lambda _: True)
    return launcher


def test_server_readiness_observes_listener_instead_of_startup_delay(launcher, clock, monkeypatch):
    connect = Mock(side_effect=[ConnectionRefusedError("starting"), ConnectionRefusedError("starting"), MagicMock()])
    monkeypatch.setattr(site_launcher.socket, "create_connection", connect)
    launcher.wait_for_server("server", timeout=1)
    assert connect.call_count == 3
    assert clock.monotonic() == pytest.approx(1000.2)


def test_poc_server_uses_provisioned_admin_port(tmp_path, monkeypatch):
    server_dir = tmp_path / "server"
    startup_dir = server_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_server.json").write_text(
        json.dumps({"servers": [{"service": {"target": "server:8002"}, "admin_port": 8002}]})
    )
    process = Mock(pid=1234)
    start = Mock(return_value=process)
    monkeypatch.setattr(poc_site_launcher, "run_command_in_subprocess", start)
    poc = POCSiteLauncher.__new__(POCSiteLauncher)
    poc.poc_dir = str(tmp_path)
    poc.server_properties = {}

    poc.start_server(0)

    assert poc.server_properties["server"].port == "8002"
    assert poc.server_properties["server"].process is process
    start.assert_called_once()
    monkeypatch.setattr(site_launcher, "process_group_alive", lambda _: True)
    connect = Mock(return_value=MagicMock())
    monkeypatch.setattr(site_launcher.socket, "create_connection", connect)
    poc.wait_for_server("server", timeout=1)
    assert connect.call_args.args[0] == ("127.0.0.1", 8002)


def test_missing_listener_fails_at_deadline_with_last_error(launcher, clock, monkeypatch):
    connect = Mock(side_effect=ConnectionRefusedError("not listening"))
    monkeypatch.setattr(site_launcher.socket, "create_connection", connect)
    with pytest.raises(RuntimeError, match="not listening.*Startup log:"):
        launcher.wait_for_server("server", timeout=0.2)
    assert clock.monotonic() == pytest.approx(1000.2)
    assert connect.call_count == 2
    assert all(call.kwargs["timeout"] > 0 for call in connect.call_args_list)


def test_dead_server_fails_before_attempting_connection(launcher, clock, monkeypatch):
    launcher.server_properties["server"].process.returncode = 7
    monkeypatch.setattr(site_launcher, "process_group_alive", lambda _: False)
    connect = Mock()
    monkeypatch.setattr(site_launcher.socket, "create_connection", connect)
    with pytest.raises(RuntimeError, match="exited before readiness.*code=7"):
        launcher.wait_for_server("server")
    connect.assert_not_called()


def test_stop_reaps_departed_leader_and_signals_remaining_owned_group(clock, monkeypatch):
    process = Mock(pid=1234, returncode=0)
    alive = [True]

    def killpg(pgid, sig):
        if sig:
            alive[0] = False
        elif not alive[0]:
            raise ProcessLookupError(pgid)

    killpg = Mock(side_effect=killpg)
    monkeypatch.setattr(utils.os, "killpg", killpg)
    utils.stop_process_group(process)
    assert call(1234, utils.signal.SIGTERM) in killpg.call_args_list
    process.poll.assert_called()


def test_command_timeout_cleans_group_and_preserves_original_failure(monkeypatch):
    process = Mock(pid=1234)
    process.wait.side_effect = subprocess.TimeoutExpired("hung command", 1)
    cleanup = Mock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(utils, "stop_process_group", cleanup)
    with pytest.raises(RuntimeError, match="hung command") as caught:
        utils.wait_command_process(process, "hung command", timeout=1)
    cleanup.assert_called_once_with(process)
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    assert "cleanup failed" in str(caught.value)


def test_missing_group_still_requires_reaping_its_leader(monkeypatch):
    process = Mock(pid=1234, returncode=None)
    monkeypatch.setattr(utils.os, "killpg", Mock(side_effect=ProcessLookupError))
    utils.stop_process_group(process, kill_timeout=5)
    process.wait.assert_called_once_with(timeout=5)


def test_nonzero_foreground_command_is_not_reported_as_success():
    command = shlex.join([sys.executable, "-c", "raise SystemExit(7)"])
    with pytest.raises(RuntimeError, match="exited with code 7"):
        utils.run_command_and_wait(command)


def test_stdin_command_timeout_reaps_its_owned_process(monkeypatch):
    started = []
    popen = utils.popen_in_new_session

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(utils, "popen_in_new_session", capture)
    command = shlex.join([sys.executable, "-c", "import time; input(); time.sleep(30)"])
    try:
        with pytest.raises(RuntimeError, match="timed out") as caught:
            utils.run_command_in_subprocess(command, stdin_data=b"go\n", timeout=0.05)
        assert len(started) == 1
        assert started[0].poll() is not None, f"timed-out command was not reaped: {caught.value}"
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_admin_login_deadline_ignores_wall_clock_adjustments(monkeypatch, clock):
    clock.time = lambda: (_ for _ in ()).throw(AssertionError("wall clock used for deadline"))
    admin = SimpleNamespace(api=SimpleNamespace(is_ready=lambda: False))
    assert utils.ensure_admin_api_logged_in(admin, timeout=0.4) is False
    assert clock.monotonic() >= 1000.4
