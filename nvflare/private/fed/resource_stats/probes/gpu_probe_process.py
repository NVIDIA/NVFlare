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

"""Own one asynchronous CUDA discovery for one execution environment."""

import atexit
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from copy import deepcopy

from ..contract import normalize_measurement
from . import gpu_nvidia

# Healthy cold initialization can take several seconds per GPU. This is a
# failure deadline, not a job-startup wait: discovery overlaps application work.
_TIMEOUT_SECONDS = 30.0
_FINAL_WAIT_SECONDS = 2.0
_OUTPUT_TIMEOUT_SECONDS = 1.0
_MAX_OUTPUT_BYTES = 64 * 1024


class GpuProbe:
    """Start once, retain the result and GPU initialization until owner cleanup.

    The owner must freeze CPU accounting before result()/close(): the helper is
    deliberately not polled or reaped during the application measurement window.
    Reuse is confined to this execution; no host or cross-worker cache is used.
    """

    def __init__(self):
        self._owner_pid = os.getpid()
        self._deadline = time.monotonic() + _TIMEOUT_SECONDS
        self._process = None
        self._value = None
        self._ready = False
        self._completed = False
        self._closed = False
        self._reaper = None
        self._result_lock = threading.Lock()
        atexit.register(self.close)
        try:
            self._process = subprocess.Popen(
                [sys.executable, gpu_nvidia.__file__, "--hold", str(self._owner_pid), str(self._deadline)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            self.close()

    def _accept_frame(self, encoded, index):
        frame = json.loads(encoded)
        if not isinstance(frame, dict) or set(frame) != {"phase", "gpu"}:
            raise ValueError("invalid GPU discovery frame")
        phase = frame["phase"]
        if phase not in ("count", "complete") or (phase == "count" and index != 1):
            raise ValueError("invalid GPU discovery phase")
        if frame["gpu"] is None:
            if phase != "complete" or self._value is not None:
                raise ValueError("missing GPU discovery count")
        else:
            value = normalize_measurement(
                "resource_capacity", {"status": "partial", "issues": ["observation_incomplete"], "gpu": frame["gpu"]}
            )["gpu"]
            total = sum(int(group["count"]) for group in value["groups"])
            if self._value is not None and total != sum(int(group["count"]) for group in self._value["groups"]):
                raise ValueError("GPU enrichment changed the verified count")
            self._value = value
        self._completed = phase == "complete"
        return self._completed

    def result(self, *, timeout: float = _FINAL_WAIT_SECONDS):
        """Wait only for a verified count; use enrichment only if already ready."""

        if os.getpid() != self._owner_pid:
            return None
        with self._result_lock:
            if self._ready:
                return deepcopy(self._value)
            self._ready = True
            if self._closed or self._process is None:
                return None
            try:
                encoded = bytearray()
                received = frames = 0
                final_deadline = time.monotonic() + timeout
                deadline = min(self._deadline, final_deadline)
                with selectors.DefaultSelector() as selector:
                    selector.register(self._process.stdout, selectors.EVENT_READ)
                    while frames < 2:
                        if b"\n" in encoded:
                            line, _, encoded = encoded.partition(b"\n")
                            if len(line) + 1 > _MAX_OUTPUT_BYTES:
                                break
                            frames += 1
                            if self._accept_frame(line, frames):
                                break
                            continue
                        if len(encoded) >= _MAX_OUTPUT_BYTES or received >= 2 * _MAX_OUTPUT_BYTES:
                            break
                        # Once count is known, optional enrichment never waits.
                        remaining = 0.0 if self._value is not None else max(0.0, deadline - time.monotonic())
                        if not selector.select(remaining):
                            break
                        chunk = os.read(self._process.stdout.fileno(), min(4096, _MAX_OUTPUT_BYTES + 1 - len(encoded)))
                        if not chunk:
                            break
                        if not received:
                            # A standalone complete frame can exceed pipe capacity.
                            # Ready output may drain within the caller's budget.
                            deadline = min(final_deadline, max(deadline, time.monotonic() + _OUTPUT_TIMEOUT_SECONDS))
                        received += len(chunk)
                        if received > 2 * _MAX_OUTPUT_BYTES:
                            break
                        encoded.extend(chunk)
            except (OSError, ValueError, TypeError, RecursionError, subprocess.SubprocessError):
                pass
            return deepcopy(self._value)

    def close(self):
        """Release the initialization hold and reap only this owner's helper."""

        if os.getpid() != self._owner_pid or self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            if not self._completed:
                # Do not add a graceful-exit wait to an unfinished discovery.
                process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            # A driver stuck in the kernel can defer even SIGKILL. Keep its
            # sole reaper alive without extending the execution's blocking
            # collection budget or handing it to an unrelated runtime owner.
            self._reaper = threading.Thread(target=self._reap, args=(process,), daemon=True, name="gpu-probe-reap")
            self._reaper.start()
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass

    @staticmethod
    def _reap(process):
        try:
            process.wait()
        except (OSError, subprocess.SubprocessError):
            pass


def probe_gpu_isolated():
    """Synchronous convenience for diagnostics; execution collectors overlap it."""

    probe = GpuProbe()
    try:
        return probe.result(timeout=_TIMEOUT_SECONDS)
    finally:
        probe.close()
