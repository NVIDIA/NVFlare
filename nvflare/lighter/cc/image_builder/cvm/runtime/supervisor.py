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
import subprocess
import sys
import time

from ..common.errors import BuildError, require
from ..common.io import write_json
from ..common.linux import run
from .audit import emit
from .gpu import readiness
from .storage import close_vault
from .systemd import notify, watchdog

PERIODIC_INTERVAL_SECONDS = 300
PERIODIC_TIMEOUT_SECONDS = 300


# A failed periodic check no longer powers the guest off at once. The workload
# is stopped and the vault closed, which drops its key from the kernel, and
# authorization is retried for this bounded window before PID 1 powers off.
QUARANTINE_WINDOW_SECONDS = 900
QUARANTINE_RETRY_SECONDS = 60
REOPEN_TIMEOUT_SECONDS = 900
WATCHDOG_SECONDS = 360
REVOCATION_TIMEOUT_SECONDS = 180


def run_reopen(timeout, environment):
    """Own the reopen process group so no unlock command survives a failed attempt."""
    try:
        with subprocess.Popen(
            [sys.executable, "-m", "cvm.runtime.bootstrap", "reopen"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        ) as child:
            try:
                output, _ = child.communicate(timeout=timeout)
            finally:
                # Kill descendants too, including cryptsetup/mount interrupted
                # by the deadline, before the parent inspects vault state.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            require(child.returncode == 0, "Vault reopen failed")
            return output
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"Vault reopen could not complete ({type(exc).__name__})") from None


def revoke_vault(config):
    """Revoke readiness and discard any mapping left by a failed reopen."""
    try:
        readiness(config, False)
    finally:
        close_vault()


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
        watchdog(REVOCATION_TIMEOUT_SECONDS)
        record.update(result="failed", finished_at=time.time(), duration_seconds=time.monotonic() - started)
        try:
            readiness(config, False)
        finally:
            write_json(state / "periodic.json", record)
        raise
    record.update(result="success", finished_at=time.time(), duration_seconds=time.monotonic() - started)
    write_json(state / "periodic.json", record)
    emit("allow")


def quarantine(config, units, state):
    """Stop the workload, close the vault, then retry authorization for a bounded window.

    No secret remains reachable while quarantined: the container and daemons are
    stopped, the vault is unmounted and its mapping closed. A successful reopen
    child re-attests, re-fetches the key and re-mounts before the units restart.
    """
    # PID 1 terminates this guest if stop/unmount or a Python operation stalls.
    # Logging has a separate bounded queue and cannot defer revocation.
    watchdog(REVOCATION_TIMEOUT_SECONDS)
    write_json(state / "quarantine.json", {"started_at": time.time(), "units": list(units)})
    run(["systemctl", "stop", *units], timeout=120)
    run(["systemctl", "stop", "docker.service", "containerd.service"], timeout=120)
    revoke_vault(config)
    emit("quarantine")
    environment = dict(os.environ)
    environment.pop("NOTIFY_SOCKET", None)
    deadline = time.monotonic() + QUARANTINE_WINDOW_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Vault quarantine deadline expired")
        watchdog(min(REOPEN_TIMEOUT_SECONDS, remaining) + 10)
        try:
            output = run_reopen(min(REOPEN_TIMEOUT_SECONDS, remaining), environment)
            require(time.monotonic() < deadline, "Vault quarantine deadline expired")
            break
        except BuildError:
            # Cleanup is deliberately outside the retried operation: failure
            # here propagates to PID 1's fail-closed power-off path.
            revoke_vault(config)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            watchdog(min(QUARANTINE_RETRY_SECONDS, remaining) + 10)
            time.sleep(min(QUARANTINE_RETRY_SECONDS, remaining))
    if output:
        print(output.decode(), end="", flush=True)
    (state / "quarantine.json").unlink(missing_ok=True)
    emit("allow")
    watchdog(WATCHDOG_SECONDS)
    run(["systemctl", "start", *units], timeout=300)


def supervise(config, units, state):
    # Consume SIGUSR1 synchronously, rather than taking locks or running a tick
    # in an asynchronous signal handler. Block it before READY so an early
    # application request is queued safely while the start transaction completes.
    signals = {signal.SIGUSR1}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        write_json(state / "periodic.json", {"sequence": 0, "result": "idle"})
        emit("allow")
        deadline = time.monotonic() + PERIODIC_INTERVAL_SECONDS
        notify("READY=1\nSTATUS=Vault authenticated; starting application services")
        watchdog(WATCHDOG_SECONDS)
        # READY completes our own start job before units depending on us start.
        run(["systemctl", "start", *units], timeout=300)
        print("CVM_WORKLOAD_STARTED", flush=True)
        sequence = 0
        while True:
            watchdog(WATCHDOG_SECONDS)
            signal.sigtimedwait(signals, max(0, deadline - time.monotonic()))
            # Include the child's runtime in the interval. An explicit request
            # starts a fresh interval; slow startup/ticks never add another full
            # sleep, and the synchronous loop never overlaps children.
            deadline = time.monotonic() + PERIODIC_INTERVAL_SECONDS
            sequence += 1
            watchdog(WATCHDOG_SECONDS)
            try:
                periodic_tick(config, state, sequence)
            except BuildError:
                quarantine(config, units, state)
                deadline = time.monotonic() + PERIODIC_INTERVAL_SECONDS
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
