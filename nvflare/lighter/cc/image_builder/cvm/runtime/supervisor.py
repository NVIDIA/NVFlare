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

"""Supervise workload startup and isolated, bounded re-attestation children."""

import os
import signal
import sys
import time

from ..common.io import write_json
from ..common.linux import run
from .audit import emit
from .gpu import readiness
from .systemd import notify

PERIODIC_INTERVAL_SECONDS = 300
PERIODIC_TIMEOUT_SECONDS = 300


def periodic_tick(config, state, sequence):
    started = time.monotonic()
    record = {"sequence": sequence, "result": "running", "started_at": time.time()}
    write_json(state / "periodic.json", record)
    try:
        # A child must not inherit permission to notify PID 1 on our behalf.
        environment = dict(os.environ)
        environment.pop("NOTIFY_SOCKET", None)
        output = run(
            [sys.executable, "-m", "cvm.runtime.bootstrap", "periodic"],
            timeout=PERIODIC_TIMEOUT_SECONDS,
            env=environment,
        )
        # The measured child prints only the sanitized audit record; command
        # outputs, keys and tokens are consumed privately by its run() calls.
        if output:
            print(output.decode(), end="", flush=True)
    except Exception:
        record.update(result="failed", finished_at=time.time(), duration_seconds=time.monotonic() - started)
        try:
            readiness(config, False)
        finally:
            write_json(state / "periodic.json", record)
        raise
    record.update(result="success", finished_at=time.time(), duration_seconds=time.monotonic() - started)
    write_json(state / "periodic.json", record)


def supervise(config, units, state):
    # Consume SIGUSR1 synchronously, rather than taking locks or running a tick
    # in an asynchronous signal handler. Block it before READY so an early
    # application request is queued safely while the start transaction completes.
    signals = {signal.SIGUSR1}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        write_json(state / "periodic.json", {"sequence": 0, "result": "idle"})
        emit("allow")
        notify("READY=1\nSTATUS=Vault authenticated; starting application services")
        # READY completes our own start job before units depending on us start.
        run(["systemctl", "start", *units], timeout=300)
        print("CVM_WORKLOAD_STARTED", flush=True)
        sequence = 0
        while True:
            signal.sigtimedwait(signals, PERIODIC_INTERVAL_SECONDS)
            sequence += 1
            periodic_tick(config, state, sequence)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
