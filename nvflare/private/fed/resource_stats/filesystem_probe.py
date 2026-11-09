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

"""Bound optional filesystem observations independently of job cleanup."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import threading
import time
from copy import deepcopy

from .handoff import _read_issues, _read_usage, _read_workspace
from .measurements import scoped_measurement

_HELPER_MODULE = "nvflare.private.fed.resource_stats.filesystem_probe_worker"
_OBSERVATION_TIMEOUT_SECONDS = 2.0
_REAP_TIMEOUT_SECONDS = 0.25
_MAX_OUTPUT_BYTES = 4096
_MEASUREMENTS = ("workspace_filesystem", "retained_content")


def _unavailable():
    return {"status": "unavailable", "issues": ["observation_incomplete"]}


def _checked_observation(name, value):
    issues = []
    measurement = scoped_measurement(name, value, issues)
    result = _read_workspace(measurement) if name == "workspace_filesystem" else _read_usage(measurement, ("bytes",))
    tags = _read_issues(issues, {name: result})
    if tags:
        result["issues"] = [item["code"] for item in tags]
    return result


def _frame_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate filesystem frame key")
        result[key] = value
    return result


class FilesystemProbe:
    """Own one helper without registering a multiprocessing shutdown join.

    The observation deadline includes interpreter startup. A completed frame is
    usable independently of the other measurement. Expired or malformed frames
    never replace a missing observation with a zero value.
    """

    def __init__(self, run_dir, *, timeout=_OBSERVATION_TIMEOUT_SECONDS):
        self._owner_pid = os.getpid()
        self._deadline = time.monotonic() + timeout
        self._process = None
        self._closed = False
        self._result = None
        self._launch_done = threading.Event()
        self._launch_lock = threading.Lock()
        try:
            # Popen can itself wait for exec/filesystem work. Keep that startup
            # off the job's finalization thread and include it in the deadline.
            threading.Thread(
                target=self._launch, args=(os.fspath(run_dir),), daemon=True, name="filesystem-probe-start"
            ).start()
        except (OSError, ValueError, RuntimeError):
            self._launch_done.set()

    def _launch(self, run_dir):
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", _HELPER_MODULE, os.fspath(run_dir), str(self._owner_pid)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            with self._launch_lock:
                if not self._closed:
                    self._process = process
                    return
            # A helper whose startup finished after abandonment is still owned.
            self._close_process(process)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        finally:
            self._launch_done.set()

    def result(self, *, timeout=None):
        """Read bounded complete frames before the absolute observation deadline."""

        if self._result is not None:
            return deepcopy(self._result)
        result = {name: _unavailable() for name in _MEASUREMENTS}
        self._result = result
        deadline = self._deadline
        if timeout is not None:
            deadline = min(deadline, time.monotonic() + timeout)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return deepcopy(result)
        self._launch_done.wait(remaining)
        if self._closed or self._process is None:
            return deepcopy(result)
        pending = bytearray()
        received = set()
        total = 0
        try:
            os.set_blocking(self._process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while len(received) < len(_MEASUREMENTS):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        break
                    if time.monotonic() >= deadline:
                        break
                    chunk = os.read(self._process.stdout.fileno(), min(1024, _MAX_OUTPUT_BYTES + 1 - total))
                    if not chunk or time.monotonic() >= deadline:
                        break
                    total += len(chunk)
                    if total > _MAX_OUTPUT_BYTES:
                        break
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, pending = pending.partition(b"\n")
                        try:
                            frame = json.loads(line, object_pairs_hook=_frame_object)
                            if not isinstance(frame, dict) or len(frame) != 1:
                                continue
                            name, value = next(iter(frame.items()))
                            if name not in _MEASUREMENTS or name in received:
                                continue
                            result[name] = _checked_observation(name, value)
                            received.add(name)
                        except (ValueError, TypeError, RecursionError):
                            continue
        except (OSError, ValueError, TypeError, RecursionError, subprocess.SubprocessError):
            pass
        return deepcopy(result)

    def close(self):
        """Kill the optional observer and bound reaping without delaying job exit."""

        with self._launch_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
        if process is None:
            return
        self._close_process(process)

    @staticmethod
    def _close_process(process):
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Kernel-uninterruptible I/O may defer even SIGKILL. Reap when the
            # kernel releases it without putting the observer on job-exit's
            # wait path. Runtime allocation cleanup remains with its owner.
            threading.Thread(
                target=FilesystemProbe._reap, args=(process,), daemon=True, name="filesystem-probe-reap"
            ).start()
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    @staticmethod
    def _reap(process):
        try:
            process.wait()
        except (OSError, subprocess.SubprocessError):
            pass


def observe_filesystem_isolated(run_dir):
    """Finish optional observations within a fixed wait plus bounded reap grace."""

    fallback = {name: _unavailable() for name in _MEASUREMENTS}
    try:
        probe = FilesystemProbe(run_dir)
    except Exception:
        return fallback
    try:
        return probe.result()
    except Exception:
        return fallback
    finally:
        try:
            probe.close()
        except Exception:
            pass  # Optional observation cleanup cannot replace the job result.
