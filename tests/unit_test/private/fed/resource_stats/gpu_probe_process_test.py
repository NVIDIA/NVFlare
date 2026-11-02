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


"""Controlled helper pipe/cleanup behavior, without processes or CUDA."""

import io
import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from nvflare.private.fed.resource_stats.probes import gpu_nvidia
from nvflare.private.fed.resource_stats.probes import gpu_probe_process as probe


def _frame(value):
    return json.dumps(value).encode() + b"\n"


class FakeSelector:
    def __init__(self, waits, ready):
        self.waits = waits
        self.ready = ready

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def register(self, *_args):
        pass

    def select(self, timeout):
        self.waits.append(timeout)
        return [True] if self.ready else []


def _helper(monkeypatch, encoded, *, ready=True):
    commands, waits, lifecycle = [], [], []
    data = io.BytesIO(encoded)
    process = SimpleNamespace(
        stdin=SimpleNamespace(close=lambda: lifecycle.append("release")),
        stdout=SimpleNamespace(fileno=lambda: 4242, close=lambda: lifecycle.append("close-output")),
        wait=lambda timeout: lifecycle.append("wait"),
        kill=lambda: lifecycle.append("kill"),
    )

    def launch(command, **kwargs):
        commands.append((command, kwargs))
        return process

    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    monkeypatch.setattr(probe.os, "read", lambda fd, size: data.read(size))
    monkeypatch.setattr(probe.selectors, "DefaultSelector", lambda: FakeSelector(waits, ready))
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: 10.0))
    return process, commands, waits, lifecycle


def test_startup_time_counts_against_remaining_deadline(monkeypatch):
    _, _, waits, _ = _helper(monkeypatch, b"", ready=False)
    session = probe.GpuProbe()
    try:
        monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: 38.0))
        assert session.result() is None
        assert waits == [2.0]
    finally:
        session.close()


def test_unresponsive_helper_is_killed_and_reaped_on_close(monkeypatch):
    process, _, _, lifecycle = _helper(monkeypatch, b"", ready=False)
    calls = []

    def wait(timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("probe", timeout)
        return -9

    process.wait = wait
    session = probe.GpuProbe()
    assert session.result() is None
    session.close()
    session.close()
    assert calls == [1.0, 1.0]
    assert lifecycle == ["release", "kill", "kill", "close-output"]


def test_forked_child_cannot_use_or_close_parent_probe(monkeypatch):
    _, _, waits, lifecycle = _helper(monkeypatch, _frame({"count": 0}))
    session = probe.GpuProbe()
    try:
        with monkeypatch.context() as forked:
            forked.setattr(probe.os, "getpid", lambda: session._owner_pid + 1)
            assert session.result() is None
            session.close()
        assert waits == lifecycle == []
    finally:
        session.close()


def test_helper_launch_failure_is_unavailable(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("cannot start helper")

    monkeypatch.setattr(probe.subprocess, "Popen", fail)
    assert probe.probe_gpu_isolated() is None


def test_finalization_does_not_wait_for_full_background_budget(monkeypatch):
    _, _, waits, _ = _helper(monkeypatch, b"", ready=False)
    session = probe.GpuProbe()
    try:
        assert session.result() is None
        assert waits == [2.0]
    finally:
        session.close()


def test_kernel_delayed_exit_retains_a_reaper_without_blocking_owner(monkeypatch):
    process, _, _, lifecycle = _helper(monkeypatch, b"", ready=False)
    entered = threading.Event()
    release = threading.Event()
    reaped = threading.Event()

    def wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("probe", timeout)
        entered.set()
        assert release.wait(10), "test did not release the blocked reaper"
        reaped.set()
        return -9

    process.wait = wait
    session = probe.GpuProbe()
    try:
        session.close()
        assert entered.wait(5)
        assert not reaped.is_set()
        assert lifecycle == ["release", "kill", "kill", "close-output"]
        release.set()
        session._reaper.join(5)
        assert not session._reaper.is_alive()
        assert reaped.is_set()
    finally:
        release.set()
        if session._reaper is not None:
            session._reaper.join(5)
        session.close()


@pytest.mark.parametrize(
    "encoded,expected,completed",
    [
        (_frame({"count": 2}), {"count": 2}, True),
        (_frame({"count": 0}), {"count": 0}, True),
        (_frame(None), None, True),
        (_frame({"count": "2"}), None, False),
        (_frame({"count": True}), None, False),
        (_frame({"count": -1}), None, False),
        (_frame({"count": 4097}), None, False),
        (_frame({"count": 1.5}), None, False),
        (_frame({"count": 2, "extra": 1}), None, False),
        (b"not-json\n", None, False),
        (_frame({"count": 0})[:-1], None, False),
        (b"x" * 129, None, False),
        (b"", None, False),
    ],
)
def test_helper_output_and_failure_semantics(monkeypatch, encoded, expected, completed):
    _, commands, waits, lifecycle = _helper(monkeypatch, encoded)
    session = probe.GpuProbe()
    try:
        assert waits == []
        assert session.result() == expected
        assert lifecycle == []  # Never reap the helper within the CPU measurement window.
        assert session.result() == expected
        command, kwargs = commands[0]
        assert command == [probe.sys.executable, gpu_nvidia.__file__, "--hold", str(probe.os.getpid()), "40.0"]
        assert "env" not in kwargs
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.DEVNULL
    finally:
        session.close()
    assert lifecycle == (
        ["release", "wait", "close-output"] if completed else ["release", "kill", "wait", "close-output"]
    )


def test_completed_result_is_reused_without_mutable_aliases(monkeypatch):
    _, commands, waits, _ = _helper(monkeypatch, _frame({"count": 2}))
    session = probe.GpuProbe()
    try:
        session.result()["count"] = 99
        assert session.result() == {"count": 2}
        assert len(commands) == len(waits) == 1
    finally:
        session.close()


def test_long_job_can_read_a_count_without_waiting(monkeypatch):
    _, _, waits, _ = _helper(monkeypatch, _frame({"count": 0}))
    session = probe.GpuProbe()
    try:
        monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: 100.0))
        assert session.result() == {"count": 0}
        assert waits == [0.0]
    finally:
        session.close()
