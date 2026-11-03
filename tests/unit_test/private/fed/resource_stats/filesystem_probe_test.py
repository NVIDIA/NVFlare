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
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.resource_stats import filesystem_probe
from nvflare.private.fed.resource_stats.collector import JobResourceCollector
from tests.timing_utils import isolate_time

UNAVAILABLE = {"status": "unavailable", "issues": ["observation_incomplete"]}
_ISOLATED_OBSERVE = filesystem_probe.observe_filesystem_isolated


class FakeSelector:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.current = b""
        self.waits = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def register(self, *args):
        pass

    def select(self, timeout):
        self.waits.append(timeout)
        self.current = next(self.chunks, b"")
        return [True] if self.current else []


def install_process(monkeypatch, chunks):
    process = SimpleNamespace(stdin=Mock(), stdout=Mock(), poll=Mock(return_value=None), kill=Mock(), wait=Mock())
    selector = FakeSelector(chunks)
    monkeypatch.setattr(filesystem_probe.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(filesystem_probe.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(filesystem_probe.os, "read", lambda *args: selector.current)
    monkeypatch.setattr(filesystem_probe.os, "set_blocking", lambda *args: None)
    return process, selector


def test_completed_measurement_survives_an_incomplete_second_frame(monkeypatch):
    process, _ = install_process(
        monkeypatch,
        [b'{"workspace_filesystem":{"capacity_bytes":1024}}\n', b'{"retained_content":'],
    )
    probe = filesystem_probe.FilesystemProbe("run")
    assert probe._launch_done.wait(1)
    try:
        result = probe.result()
        assert result == {"workspace_filesystem": {"capacity_bytes": 1024}, "retained_content": UNAVAILABLE}
        assert probe.result() == result
    finally:
        probe.close()
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=filesystem_probe._REAP_TIMEOUT_SECONDS)


def test_safely_reported_partial_subtotal_is_preserved(monkeypatch):
    retained = {"status": "partial", "issues": ["observation_incomplete"], "bytes": 7}
    encoded = json.dumps({"retained_content": retained}).encode() + b"\n"
    install_process(monkeypatch, [encoded])
    probe = filesystem_probe.FilesystemProbe("run")
    assert probe._launch_done.wait(1)
    try:
        assert probe.result()["retained_content"] == retained
    finally:
        probe.close()


def test_startup_counts_against_observation_deadline(monkeypatch):
    process, selector = install_process(monkeypatch, [])
    clock = isolate_time(monkeypatch, filesystem_probe)
    clock.monotonic = Mock(side_effect=[10.0, 13.0])
    probe = filesystem_probe.FilesystemProbe("run", timeout=2.0)
    assert probe._launch_done.wait(1)
    try:
        assert probe.result() == {name: UNAVAILABLE for name in filesystem_probe._MEASUREMENTS}
        assert selector.waits == []
    finally:
        probe.close()
    process.kill.assert_called_once()


@pytest.mark.parametrize("ticks", [(0.0, 0.25, 0.5, 2.0), (0.0, 0.25, 0.5, 0.75, 2.0)])
def test_result_after_deadline_is_not_accepted(monkeypatch, ticks):
    _, selector = install_process(monkeypatch, [b'{"retained_content":{"bytes":123}}\n'])
    clock = isolate_time(monkeypatch, filesystem_probe)
    clock.monotonic = Mock(side_effect=ticks)
    probe = filesystem_probe.FilesystemProbe("run", timeout=1.0)
    assert probe._launch_done.wait(1)
    try:
        assert probe.result()["retained_content"] == UNAVAILABLE
        assert selector.waits == [0.5]
    finally:
        probe.close()


@pytest.mark.parametrize(
    "encoded",
    [b'{"retained_content":{"bytes":"NaN"}}\n', b"x" * (filesystem_probe._MAX_OUTPUT_BYTES + 1)],
)
def test_invalid_or_oversized_output_never_becomes_a_measurement(monkeypatch, encoded):
    install_process(monkeypatch, [encoded])
    probe = filesystem_probe.FilesystemProbe("run")
    assert probe._launch_done.wait(1)
    try:
        assert probe.result()["retained_content"] == UNAVAILABLE
    finally:
        probe.close()


def test_blocked_reap_is_delegated_to_daemon_owner(monkeypatch):
    process, _ = install_process(monkeypatch, [])
    process.wait.side_effect = subprocess.TimeoutExpired("helper", filesystem_probe._REAP_TIMEOUT_SECONDS)
    reaper = Mock()
    thread = Mock(return_value=reaper)
    probe = filesystem_probe.FilesystemProbe("run")
    assert probe._launch_done.wait(1)
    monkeypatch.setattr(filesystem_probe.threading, "Thread", thread)
    probe.close()
    probe.close()
    assert thread.call_args.kwargs["daemon"] is True
    assert thread.call_args.kwargs["args"] == (process,)
    reaper.start.assert_called_once()
    process.wait.assert_called_once()


def test_observer_setup_failure_reports_unavailable(monkeypatch):
    monkeypatch.setattr(filesystem_probe, "FilesystemProbe", Mock(side_effect=RuntimeError("cannot start observer")))
    # The autouse collector fixture replaces the convenience function, so call
    # the original definition through the module constant saved before fixtures run.
    assert _ISOLATED_OBSERVE("run") == {name: UNAVAILABLE for name in filesystem_probe._MEASUREMENTS}


def test_blocked_startup_is_abandoned_and_late_process_is_still_cleaned_up(monkeypatch):
    process, _ = install_process(monkeypatch, [])
    entered, release = threading.Event(), threading.Event()

    def start(*args, **kwargs):
        entered.set()
        release.wait(2)
        return process

    monkeypatch.setattr(filesystem_probe.subprocess, "Popen", start)
    probe = filesystem_probe.FilesystemProbe("run", timeout=0.03)
    try:
        assert entered.wait(1)
        start_time = time.monotonic()
        assert probe.result() == {name: UNAVAILABLE for name in filesystem_probe._MEASUREMENTS}
        probe.close()
        assert time.monotonic() - start_time < 0.5
        assert not release.is_set()
    finally:
        release.set()
        assert probe._launch_done.wait(1)
        probe.close()
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=filesystem_probe._REAP_TIMEOUT_SECONDS)


@pytest.mark.parametrize(
    "bad_frame",
    [
        b"not JSON\n",
        b'{"retained_content":{"bytes":1,"bytes":2}}\n',
        b'{"retained_content":{"bytes":false}}\n',
        b'{"retained_content":{"bytes":NaN}}\n',
        b'{"retained_content":{"status":"partial","bytes":1}}\n',
        b'{"retained_content":{"status":"unavailable","issues":["invented"]}}\n',
    ],
)
def test_malformed_frame_does_not_erase_an_independent_completed_observation(monkeypatch, bad_frame):
    install_process(monkeypatch, [bad_frame + b'{"workspace_filesystem":{"capacity_bytes":1024}}\n'])
    probe = filesystem_probe.FilesystemProbe("run")
    try:
        assert probe.result() == {"workspace_filesystem": {"capacity_bytes": 1024}, "retained_content": UNAVAILABLE}
    finally:
        probe.close()


def test_duplicate_frames_and_oversized_tail_do_not_replace_completed_values(monkeypatch):
    install_process(
        monkeypatch,
        [
            b'{"workspace_filesystem":{"capacity_bytes":1024}}\n',
            b'{"workspace_filesystem":{"capacity_bytes":2048}}\n',
            b"x" * filesystem_probe._MAX_OUTPUT_BYTES,
        ],
    )
    probe = filesystem_probe.FilesystemProbe("run")
    try:
        assert probe.result() == {"workspace_filesystem": {"capacity_bytes": 1024}, "retained_content": UNAVAILABLE}
    finally:
        probe.close()


def test_execution_clock_and_cpu_freeze_before_starting_the_single_observer(monkeypatch, tmp_path):
    ticks = iter((0, 1_000_000_000))
    job = JobResourceCollector(
        tmp_path,
        clock_ns=lambda: next(ticks),
        capacity_probe=lambda: {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": 0}},
    )
    observed = []

    def observe(run_dir):
        assert job._snapshot["duration_seconds"] == 1
        assert "cpu_seconds" in job._snapshot["cpu_consumed"]
        observed.append(run_dir)
        return {"workspace_filesystem": {"capacity_bytes": 4096}, "retained_content": {"bytes": 7}}

    monkeypatch.setattr(filesystem_probe, "observe_filesystem_isolated", observe)
    result = job.finish(message_traffic={"sent_bytes": 0, "received_bytes": 0})
    assert observed == [tmp_path]
    assert result["resource_utilization"]["retained_content"] == {"bytes": 7}
    assert result["issues"] == []
