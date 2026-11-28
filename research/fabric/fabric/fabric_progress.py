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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Log-friendly progress/ETA; observes iteration without touching data or RNG."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    """Newline output survives FLARE's subprocess logging and redirected stdout."""

    def __init__(self, total: int, label: str, log_path: Path, *, unit: str = "batch"):
        self.total = total
        self.label = label
        self.log_path = log_path
        self.unit = unit
        self.completed = 0
        self.started = time.perf_counter()
        self.last_print = float("-inf")
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def show(self, detail: str = "", *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_print < 10.0:
            return
        elapsed = now - self.started
        remaining = elapsed / self.completed * (self.total - self.completed) if self.completed else None
        percent = 100.0 * self.completed / self.total if self.total else 0.0
        line = (
            f"[FABRIC progress] {self.label} {detail} "
            f"{self.unit}={self.completed}/{self.total} ({percent:.1f}%) "
            f"elapsed={duration(elapsed)} ETA={duration(remaining)}"
        )
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.last_print = now

    def advance(self, detail: str = "", *, force: bool = False) -> None:
        self.completed += 1
        self.show(detail, force=force or self.completed in (1, self.total))


class ProgressLoader:
    """Keep one original loader and yield each original batch exactly once."""

    def __init__(self, loader, epochs: int, label: str, log_path: Path):
        self.loader = loader
        self.epochs = epochs
        self.label = label
        self.log_path = log_path
        self.epoch = 0
        self.progress = None

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        self.epoch += 1
        batches = len(self.loader)
        if self.progress is None:
            self.progress = Progress(batches * self.epochs, self.label, self.log_path)
        detail = f"epoch={self.epoch}/{self.epochs} epoch_batch=0/{batches}"
        self.progress.show(detail, force=True)
        completed = 0
        exhausted = False
        try:
            for batch in self.loader:
                yield batch
                # Count only after the caller finishes processing this batch.
                completed += 1
                detail = f"epoch={self.epoch}/{self.epochs} epoch_batch={completed}/{batches}"
                self.progress.advance(detail, force=completed == batches)
            exhausted = True
        finally:
            if not exhausted:
                self.progress.show(detail + " stopped_before_epoch_complete", force=True)


@contextmanager
def loader_progress(module, *, epochs: int, label: str, log_path: Path):
    """Temporarily observe the factory used by an unchanged core function.

    Each client task runs in its own process; evaluation runs after clients stop.
    Do not use this process-local wrapper for concurrent calls to the same module.
    """
    original = module.make_loader

    @wraps(original)
    def make_loader(*args, **kwargs):
        return ProgressLoader(original(*args, **kwargs), epochs, label, log_path)

    module.make_loader = make_loader
    try:
        yield
    finally:
        module.make_loader = original
