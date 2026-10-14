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

"""Failure-path tests for the isolated CVM acceptance runner."""

import importlib.util
import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from cvm.common.errors import BuildError

LAB = Path(__file__).resolve().parents[4] / "integration_test/lighter/cc/image_builder"


def load_runner():
    spec = importlib.util.spec_from_file_location("acceptance_runner", LAB / "acceptance_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode", [1, -signal.SIGKILL])
def test_failed_hardware_suite_cleans_resources_and_reports_exit_status(tmp_path, returncode):
    runner = load_runner()
    process = Mock(pid=1234)
    process.wait.return_value = returncode
    process.poll.return_value = returncode
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner, "_shutdown_vaults") as shutdown_vaults,
        patch.object(runner, "_cleanup_suite_firewall") as cleanup_firewall,
        pytest.raises(BuildError, match=f"exit status {returncode}.*inspect"),
    ):
        runner.run_suite("test_hardware.py", {}, tmp_path)
    shutdown_vaults.assert_called_once_with(tmp_path)
    cleanup_firewall.assert_called_once_with(process.pid)
    process.wait.assert_called_once_with()


def test_successful_hardware_suite_cleans_firewall_without_stopping_vaults(tmp_path):
    runner = load_runner()
    process = Mock(pid=1234)
    process.wait.return_value = 0
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner, "_shutdown_vaults") as shutdown_vaults,
        patch.object(runner, "_cleanup_suite_firewall") as cleanup_firewall,
    ):
        runner.run_suite("test_hardware.py", {}, tmp_path)
    shutdown_vaults.assert_not_called()
    cleanup_firewall.assert_called_once_with(process.pid)


def test_http_suite_never_touches_hardware_firewall(tmp_path):
    runner = load_runner()
    process = Mock(pid=1234)
    process.wait.return_value = 1
    process.poll.return_value = 1
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner, "_shutdown_vaults") as shutdown_vaults,
        patch.object(runner, "_cleanup_suite_firewall") as cleanup_firewall,
        pytest.raises(BuildError, match="exit status 1"),
    ):
        runner.run_suite("test_http.py", {}, tmp_path)
    shutdown_vaults.assert_called_once_with(tmp_path)
    cleanup_firewall.assert_not_called()


def test_forced_stop_reaps_suite_before_vault_cleanup(tmp_path):
    runner = load_runner()
    events = []
    process = Mock(pid=1234)
    process.poll.return_value = None

    def wait(timeout=None):
        events.append(("wait", timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired("suite", timeout)
        return -signal.SIGKILL

    process.wait.side_effect = wait
    with (
        patch.object(runner.os, "killpg", side_effect=lambda pid, sig: events.append(("killpg", pid, sig))),
        patch.object(runner, "_shutdown_vaults", side_effect=lambda output: events.append(("vaults", output))),
    ):
        runner._stop_suite(process, tmp_path)
    assert events == [
        ("killpg", 1234, signal.SIGINT),
        ("wait", runner.SUITE_SHUTDOWN_SECONDS),
        ("killpg", 1234, signal.SIGKILL),
        ("wait", None),
        ("vaults", tmp_path),
    ]


def test_interrupted_suite_forces_reap_before_vault_and_firewall_cleanup(tmp_path):
    runner = load_runner()
    events = []
    process = Mock(pid=1234)
    process.poll.return_value = None

    def wait(timeout=None):
        events.append(("wait", timeout))
        if timeout is None and events.count(("wait", None)) == 1:
            raise runner._SuiteSignal(signal.SIGTERM)
        if timeout is not None:
            raise subprocess.TimeoutExpired("suite", timeout)
        return -signal.SIGKILL

    process.wait.side_effect = wait
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner.os, "killpg", side_effect=lambda pid, sig: events.append(("killpg", pid, sig))),
        patch.object(runner, "_shutdown_vaults", side_effect=lambda output: events.append(("vaults", output))),
        patch.object(runner, "_cleanup_suite_firewall", side_effect=lambda pid: events.append(("firewall", pid))),
        pytest.raises(BuildError, match="interrupted by SIGTERM"),
    ):
        runner.run_suite("test_hardware.py", {}, tmp_path)
    assert events == [
        ("wait", None),
        ("killpg", 1234, signal.SIGINT),
        ("wait", runner.SUITE_SHUTDOWN_SECONDS),
        ("killpg", 1234, signal.SIGKILL),
        ("wait", None),
        ("vaults", tmp_path),
        ("firewall", 1234),
    ]


def test_vault_cleanup_error_still_cleans_firewall_and_retains_exit_status(tmp_path):
    runner = load_runner()
    process = Mock(pid=1234)
    process.wait.return_value = 2
    process.poll.return_value = 2
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner, "_shutdown_vaults", side_effect=BuildError("vault locked")),
        patch.object(runner, "_cleanup_suite_firewall") as cleanup_firewall,
        pytest.raises(BuildError) as error,
    ):
        runner.run_suite("test_hardware.py", {}, tmp_path)
    assert "exit status 2" in str(error.value)
    assert "suite resources: vault locked" in str(error.value)
    assert str(tmp_path / "test_hardware.log") in str(error.value)
    cleanup_firewall.assert_called_once_with(process.pid)


def test_firewall_cleanup_error_is_reported_with_suite_failure(tmp_path):
    runner = load_runner()
    process = Mock(pid=1234)
    process.wait.return_value = 3
    process.poll.return_value = 3
    deletion_error = subprocess.CalledProcessError(1, ["nft", "delete", "table"])
    with (
        patch.object(runner.subprocess, "Popen", return_value=process),
        patch.object(runner.signal, "signal", return_value=signal.SIG_DFL),
        patch.object(runner, "_shutdown_vaults"),
        patch.object(runner, "_cleanup_suite_firewall", side_effect=deletion_error),
        pytest.raises(BuildError) as error,
    ):
        runner.run_suite("test_hardware.py", {}, tmp_path)
    assert "exit status 3" in str(error.value)
    assert "host firewall:" in str(error.value)
    assert str(tmp_path / "test_hardware.log") in str(error.value)


def test_firewall_cleanup_deletes_only_exact_inet_suite_table():
    runner = load_runner()
    listing = {
        "nftables": [
            {"table": {"family": "inet", "name": "cvm_acceptance_12345"}},
            {"table": {"family": "ip", "name": "cvm_acceptance_1234"}},
            {"table": {"family": "inet", "name": "other"}},
            {"table": {"family": "inet", "name": "cvm_acceptance_1234"}},
        ]
    }
    with patch.object(runner.subprocess, "run", side_effect=[Mock(stdout=json.dumps(listing).encode()), Mock()]) as run:
        runner._cleanup_suite_firewall(1234)
    assert run.call_args_list == [
        call(["nft", "-j", "list", "tables"], capture_output=True, check=True, timeout=10),
        call(["nft", "delete", "table", "inet", "cvm_acceptance_1234"], capture_output=True, check=True, timeout=10),
    ]


def test_firewall_cleanup_skips_absent_table():
    runner = load_runner()
    listing = {"nftables": [{"table": {"family": "inet", "name": "other"}}]}
    with patch.object(runner.subprocess, "run", return_value=Mock(stdout=json.dumps(listing).encode())) as run:
        runner._cleanup_suite_firewall(1234)
    run.assert_called_once_with(["nft", "-j", "list", "tables"], capture_output=True, check=True, timeout=10)


@pytest.mark.parametrize("failed_command", ["list", "delete"])
def test_firewall_cleanup_surfaces_nft_failures(failed_command):
    runner = load_runner()
    listing = {"nftables": [{"table": {"family": "inet", "name": "cvm_acceptance_1234"}}]}
    error = subprocess.CalledProcessError(1, ["nft", failed_command])
    results = [error] if failed_command == "list" else [Mock(stdout=json.dumps(listing).encode()), error]
    with patch.object(runner.subprocess, "run", side_effect=results) as run:
        with pytest.raises(subprocess.CalledProcessError):
            runner._cleanup_suite_firewall(1234)
    assert run.call_count == (1 if failed_command == "list" else 2)
