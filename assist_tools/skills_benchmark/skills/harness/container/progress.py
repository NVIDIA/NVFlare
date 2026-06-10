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

"""Progress heartbeat writer for in-container benchmark runs."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def epoch_seconds() -> int:
    return int(time.time())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ProgressWriter:
    def __init__(self, mode: str, script_start_epoch: int, progress_log: Path):
        self.mode = mode
        self.script_start_epoch = script_start_epoch
        self.progress_log = progress_log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def write(self, phase: str, status: str, epoch: int | None = None) -> None:
        epoch = epoch_seconds() if epoch is None else epoch
        elapsed = epoch - self.script_start_epoch
        timestamp = utc_timestamp()
        print(
            f"[{timestamp}] benchmark progress: mode={self.mode} phase={phase} "
            f"status={status} elapsed_seconds={elapsed}",
            file=sys.stderr,
            flush=True,
        )
        with self._lock:
            with self.progress_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "timestamp": timestamp,
                            "mode": self.mode,
                            "phase": phase,
                            "status": status,
                            "elapsed_seconds": elapsed,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def start_heartbeat(self, phase: str, interval_seconds: int) -> None:
        if interval_seconds <= 0:
            return

        def loop() -> None:
            while not self._stop.wait(interval_seconds):
                self.write(phase, "running")

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
