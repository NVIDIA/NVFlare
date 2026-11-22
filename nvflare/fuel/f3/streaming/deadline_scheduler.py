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

"""Process-local deadlines for short, non-I/O receive-task cleanup callbacks."""

import heapq
import itertools
import logging
import math
import os
import threading
import time
import weakref

log = logging.getLogger(__name__)
_schedulers = weakref.WeakSet()


def _after_fork():
    # Only the forking thread survives. Never acquire a queue lock that may
    # have belonged to a vanished worker; keep the child's copy of its queue.
    for scheduler in list(_schedulers):
        scheduler._condition = threading.Condition()
        scheduler._thread = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class _Deadline:
    def __init__(self, scheduler, interval, callback):
        self.scheduler = scheduler
        self.interval = interval
        self.callback = callback

    def cancel(self):
        self.scheduler.cancel(self)


class DeadlineScheduler:
    """Run cleanup deadlines on one lazily started daemon thread.

    Cancellation releases the callback immediately. An already claimed callback
    may finish; callers must make cleanup idempotent. Callbacks run outside the
    queue lock and must not perform I/O or wait on application work.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._queue = []
        self._sequence = itertools.count()
        self._thread = None
        self._stopped = False
        self._pending = 0
        self._expired = 0
        self._cancelled = 0
        self._max_cleanup_lag = 0.0
        _schedulers.add(self)

    def schedule(self, interval, callback):
        """Schedule a finite nonnegative interval, or return None after shutdown.

        Invalid intervals/deadlines raise ValueError before work is admitted.
        """
        with self._condition:
            if self._stopped:
                return None
            try:
                if interval < 0 or not math.isfinite(interval):
                    raise ValueError
                deadline = time.monotonic() + interval
                if not math.isfinite(deadline):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                raise ValueError("deadline interval must be finite and nonnegative") from None
            if self._thread is None or not self._thread.is_alive():
                worker = threading.Thread(target=self._run, name="f3-rx-cleanup", daemon=True)
                # Do not leave queued work or a phantom worker if start fails.
                worker.start()
                self._thread = worker
            entry = _Deadline(self, interval, callback)
            heapq.heappush(self._queue, (deadline, next(self._sequence), entry))
            self._pending += 1
            self._condition.notify()
            return entry

    def cancel(self, entry):
        with self._condition:
            if entry.callback is None:
                return
            entry.callback = None
            self._pending -= 1
            self._cancelled += 1
            # Cancelled entries retain no task, and periodic compaction also
            # bounds their bookkeeping when an earlier deadline remains live.
            if len(self._queue) > max(64, 2 * self._pending):
                self._queue = [item for item in self._queue if item[2].callback is not None]
                heapq.heapify(self._queue)
            self._condition.notify()

    def snapshot(self):
        """Return queue gauges and lifetime counters (lag is in seconds)."""
        with self._condition:
            return {
                "pending": self._pending,
                "queue_depth": len(self._queue),
                "expired": self._expired,
                "cancelled": self._cancelled,
                "max_cleanup_lag": self._max_cleanup_lag,
                "worker_alive": self._thread is not None and self._thread.is_alive(),
                "stopped": self._stopped,
            }

    def shutdown(self):
        """Reject new deadlines, release queued callbacks and join the worker."""
        with self._condition:
            self._stopped = True
            for _, _, entry in self._queue:
                entry.callback = None
            self._cancelled += self._pending
            self._pending = 0
            self._queue.clear()
            worker = self._thread
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def _run(self):
        while True:
            with self._condition:
                while True:
                    if self._stopped:
                        return
                    while self._queue and self._queue[0][2].callback is None:
                        heapq.heappop(self._queue)
                    if not self._queue:
                        self._condition.wait()
                        continue
                    deadline, _, entry = self._queue[0]
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        # Large finite local retry windows can exceed the OS
                        # wait limit. Wake in bounded steps without shortening
                        # their actual deadline or stranding unrelated cleanup.
                        self._condition.wait(min(delay, threading.TIMEOUT_MAX))
                        continue
                    heapq.heappop(self._queue)
                    callback = entry.callback
                    entry.callback = None
                    self._pending -= 1
                    self._max_cleanup_lag = max(self._max_cleanup_lag, -delay)
                    break
            try:
                callback()
            except Exception:
                # One broken cleanup must not strand every subsequent task.
                log.error("Receive-task deadline cleanup failed")
            finally:
                callback = None
                entry = None
                with self._condition:
                    self._expired += 1
