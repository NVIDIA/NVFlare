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

from ..common.contracts import (
    CPU_ATTESTATION_TIMEOUT_SECONDS,
    GPU_ATTESTATION_TIMEOUT_SECONDS,
    PROOF_EXPIRY_MARGIN_SECONDS,
)
from ..common.errors import BuildError, require
from ..common.io import read_json, write_json
from ..common.linux import run
from .audit import emit, emit_diagnostic, flush
from .gpu import confirm_not_ready, readiness
from .storage import close_vault
from .systemd import notify, watchdog

# Participants that do not expose an application proof keep the original
# periodic integrity cadence. Proof-producing participants schedule from the
# authenticated token expiration recorded in protected /run/cvm state.
PERIODIC_INTERVAL_SECONDS = 240
CPU_PERIODIC_TIMEOUT_SECONDS = CPU_ATTESTATION_TIMEOUT_SECONDS
GPU_PERIODIC_TIMEOUT_SECONDS = GPU_ATTESTATION_TIMEOUT_SECONDS


# A failed periodic check stops the workload and closes the vault, which drops
# its key from the kernel. CPU-only guests retry authorization for this bounded
# window. Revoking GPU Ready state is terminal for the current guest, so GPU
# guests power off after containment instead of entering this retry window.
QUARANTINE_WINDOW_SECONDS = 900
QUARANTINE_RETRY_SECONDS = 60
REOPEN_TIMEOUT_SECONDS = 900
WATCHDOG_SECONDS = 360
REVOCATION_TIMEOUT_SECONDS = 180
TERMINAL_AUDIT_FLUSH_SECONDS = 1


def periodic_timeout(config):
    return GPU_PERIODIC_TIMEOUT_SECONDS if config.get("gpu") == "nvidia_cc" else CPU_PERIODIC_TIMEOUT_SECONDS


def next_periodic_deadline(config, state, fallback=None):
    """Return a monotonic deadline that leaves time to renew before proof expiry."""
    proof = state / "application-proof.json"
    if not proof.is_file():
        return time.monotonic() + PERIODIC_INTERVAL_SECONDS if fallback is None else fallback
    expires_at = read_json(proof).get("expires_at")
    require(type(expires_at) in (int, float), "Invalid application proof expiration state")
    delay = expires_at - time.time() - periodic_timeout(config) - PROOF_EXPIRY_MARGIN_SECONDS
    require(delay > 0, "Application proof cannot be renewed before expiration")
    return time.monotonic() + delay


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


def revoke_vault(config, *, tolerate_readiness_failure=False):
    """Close the vault; a failed GPU reset requires explicit Not Ready readback."""
    confirmed_after_error = False
    try:
        try:
            readiness(config, False)
        except BuildError:
            if not tolerate_readiness_failure:
                raise
            confirm_not_ready(config)
            confirmed_after_error = True
    finally:
        close_vault()
    return confirmed_after_error


def best_effort_revoke(config):
    """Try to remove readiness and the key without masking a terminal failure."""
    try:
        watchdog(REVOCATION_TIMEOUT_SECONDS)
    except Exception:
        pass
    try:
        revoke_vault(config)
    except Exception:
        pass


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
            timeout=periodic_timeout(config),
            env=environment,
        )
        # The measured child prints only the sanitized audit record; command
        # outputs, keys and tokens are consumed privately by its run() calls.
        if output:
            print(output.decode(), end="", flush=True)
    except Exception:
        watchdog(REVOCATION_TIMEOUT_SECONDS)
        record.update(result="failed", finished_at=time.time(), duration_seconds=time.monotonic() - started)
        # Quarantine stops the application units before clearing GPU Ready
        # state. Revoking readiness while a CUDA workload is still running can
        # make cvm_app.service fail and trigger its forced-poweroff policy
        # before the guest has entered quarantine.
        write_json(state / "periodic.json", record)
        raise
    record.update(result="success", finished_at=time.time(), duration_seconds=time.monotonic() - started)
    write_json(state / "periodic.json", record)
    emit("allow")


def quarantine(config, units, state):
    """Stop the workload and close the vault; recover only CPU-only guests.

    The container and daemons are stopped, GPU readiness is revoked, and the
    vault is unmounted with its mapping closed. Revoking Ready state requires a
    GPU guest restart, so GPU quarantine returns without attempting reopen. A
    successful CPU reopen child re-attests, re-fetches the key and re-mounts
    before the units restart.
    """
    diagnostic_emitted = False

    def revoke():
        nonlocal diagnostic_emitted
        if revoke_vault(config, tolerate_readiness_failure=True) and not diagnostic_emitted:
            diagnostic_emitted = True
            emit_diagnostic("gpu_readiness_revoke_failed")

    # PID 1 terminates this guest if stop/unmount or a Python operation stalls.
    # Logging has a separate bounded queue and cannot defer revocation.
    try:
        watchdog(REVOCATION_TIMEOUT_SECONDS)
        write_json(state / "quarantine.json", {"started_at": time.time(), "units": list(units)})
        run(["systemctl", "stop", *units], timeout=120)
        run(["systemctl", "stop", "docker.service", "containerd.service"], timeout=120)
    except Exception:
        # PID 1 still forces poweroff. Try to remove GPU Ready and the vault key
        # first, while preserving the stop/configuration error as the cause.
        best_effort_revoke(config)
        raise
    revoke()
    emit("quarantine")
    if config.get("gpu") == "nvidia_cc":
        emit_diagnostic("gpu_quarantine_terminal")
        # The supervisor exits immediately after terminal containment. Give
        # the daemon writer time to publish the decision and diagnostic, but
        # never let untrusted audit storage delay fail-closed poweroff.
        flush(TERMINAL_AUDIT_FLUSH_SECONDS)
        return
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
            # Give cleanup its own bounded phase, including after a reopen
            # consumed the entire retry window. Vault closure and an explicit
            # GPU Not Ready state are required before authorization can retry.
            watchdog(REVOCATION_TIMEOUT_SECONDS)
            revoke()
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
        deadline = next_periodic_deadline(config, state)
        notify("READY=1\nSTATUS=Vault authenticated; starting application services")
        watchdog(WATCHDOG_SECONDS)
        # READY completes our own start job before units depending on us start.
        run(["systemctl", "start", *units], timeout=300)
        print("CVM_WORKLOAD_STARTED", flush=True)
        sequence = 0
        while True:
            watchdog(WATCHDOG_SECONDS)
            signal.sigtimedwait(signals, max(0, deadline - time.monotonic()))
            # For workloads without an application proof, retain the original
            # start-to-start cadence. A signal starts a fresh interval and an
            # overrun triggers the next check immediately.
            fallback = time.monotonic() + PERIODIC_INTERVAL_SECONDS
            sequence += 1
            watchdog(WATCHDOG_SECONDS)
            try:
                periodic_tick(config, state, sequence)
                deadline = next_periodic_deadline(config, state, fallback)
            except BuildError as failure:
                try:
                    quarantine(config, units, state)
                except Exception:
                    # Containment or CPU recovery may have failed partway
                    # through. CPU recovery can reopen access, so revoke it
                    # again before PID 1's forced-poweroff path takes over.
                    # GPU quarantine already performed its one terminal Ready
                    # transition and attempted vault closure on every path.
                    if config.get("gpu") != "nvidia_cc":
                        best_effort_revoke(config)
                    raise
                if config.get("gpu") == "nvidia_cc":
                    # GPU Ready-state revocation is intentionally terminal.
                    # Re-raise the appraisal failure so systemd powers off the
                    # contained guest without an impossible reopen loop or a
                    # second readiness transition.
                    raise failure
                try:
                    deadline = next_periodic_deadline(config, state)
                except Exception:
                    best_effort_revoke(config)
                    raise
            except Exception:
                # Internal failures do not enter the recovery loop. Revoke what
                # we can before this service failure makes PID 1 power off.
                best_effort_revoke(config)
                raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
