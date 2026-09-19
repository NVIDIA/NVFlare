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

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


def wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Supervisor did not reach the expected state")


@pytest.fixture
def supervisor(tmp_path):
    source = tmp_path / "source"
    (source / "startup").mkdir(parents=True)
    (source / "local").mkdir()
    template = Path(__file__).resolve().parents[3] / "nvflare/lighter/templates/master_template.yml"
    script = yaml.safe_load(template.read_text())["sub_start_sh"]
    for key, value in {
        "type": "client",
        "app_name": "client_train",
        "cln_uid": "uid=site-1",
        "org_name": "org",
        "config_folder": "",
    }.items():
        script = script.replace("{~~" + key + "~~}", value)
    (source / "startup/sub_start.sh").write_text(script)
    tools = tmp_path / "bin"
    tools.mkdir()
    child = tools / "python3"
    child.write_text(
        f"#!{sys.executable}\n"
        + """import os, signal, sys, time
from pathlib import Path
events = Path(os.environ["EVENTS"])
def stop(*args):
    with events.open("a") as stream:
        stream.write(f"stop {os.getpid()}\\n")
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if os.environ.get("IGNORE_TERM") else stop)
with events.open("a") as stream:
    stream.write(f"start {os.getpid()}\\n")
if os.environ.get("FAIL_FAST"):
    sys.exit(1)
while True:
    time.sleep(0.05)
"""
    )
    child.chmod(0o755)
    processes = []
    events = tmp_path / "events"
    runtime = tmp_path / "runtime"

    def launch(fail_fast=False, stale_pid=False, ignore_term=False):
        env = dict(
            os.environ,
            PATH=str(tools) + os.pathsep + os.environ["PATH"],
            EVENTS=str(events),
            NVFL_WORKSPACE=str(runtime),
        )
        if fail_fast:
            env["FAIL_FAST"] = "1"
        if ignore_term:
            env["IGNORE_TERM"] = "1"
        log = (tmp_path / f"log-{len(processes)}").open("w")
        argv = ["bash", str(source / "startup/sub_start.sh"), "--foreground"]
        if stale_pid:
            runtime.mkdir(exist_ok=True)
            (runtime / "pid.fl").write_text("99999999")
            (runtime / "shutdown.fl").touch()
            argv = [
                "bash",
                "-c",
                'echo $$ > "$NVFL_WORKSPACE/daemon_pid.fl"; exec bash "$1" --foreground',
                "supervisor",
                str(source / "startup/sub_start.sh"),
            ]
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        processes.append(process)
        return process

    yield launch, runtime, source, events
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@pytest.mark.parametrize("action", ["signal", "shutdown", "restart"])
def test_foreground_signal_shutdown_and_restart(supervisor, action):
    launch, runtime, source, events = supervisor
    process = launch()
    wait_for(lambda: events.exists() and "start" in events.read_text())
    assert process.poll() is None
    if action == "restart":
        (runtime / "restart.fl").touch()
        wait_for(lambda: events.read_text().count("start") == 2)
        assert "stop" in events.read_text()
        assert process.poll() is None
    if action == "shutdown":
        (runtime / "shutdown.fl").touch()
    else:
        process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=15) == 0
    assert events.read_text().count("stop") == events.read_text().count("start")
    assert not (runtime / "daemon_pid.fl").exists()
    assert not (source / "pid.fl").exists()
    assert not (source / "transfer").exists()


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_foreground_stop_leaves_time_for_cleanup_when_child_ignores_term(supervisor, stop_signal):
    launch, runtime, source, events = supervisor
    process = launch(ignore_term=True)
    wait_for(lambda: events.exists() and "start" in events.read_text())
    child_pid = int(events.read_text().split()[1])
    process.send_signal(stop_signal)
    # Leave margin for cleanup before a runtime's 10-second stop deadline.
    assert process.wait(timeout=8) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    for marker in ("pid.fl", "daemon_pid.fl", "shutdown.fl", "restart.fl"):
        assert not (runtime / marker).exists()


def test_foreground_crash_loop_exits_nonzero(supervisor):
    launch, runtime, source, events = supervisor
    process = launch(fail_fast=True)
    assert process.wait(timeout=15) == 1
    assert events.read_text().count("start") == 5
    assert not (runtime / "daemon_pid.fl").exists()


def test_foreground_reused_container_pid_does_not_block_boot(supervisor):
    launch, runtime, source, events = supervisor
    process = launch(stale_pid=True)
    wait_for(lambda: events.exists() and "start" in events.read_text())
    assert process.poll() is None
    assert not (runtime / "shutdown.fl").exists()
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=15) == 0
