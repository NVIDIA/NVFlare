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

"""Bound optional statistics I/O without joining blocked operations at shutdown."""

import os
import threading

_WAIT_SECONDS = 0.05
_busy = threading.Lock()


def _reset_after_fork():
    global _busy
    _busy = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def optional_io(operation, *args, default=None, **kwargs):
    """Wait at most 50 ms; skip while busy. A timeout does not cancel I/O.

    Keep lifecycle state changes outside the callback. Adopt its result on
    the calling thread, since a timed-out callback may finish later.
    """
    busy = _busy
    if not busy.acquire(blocking=False):
        return default
    done = threading.Event()
    result = [default]

    def run():
        try:
            result[0] = operation(*args, **kwargs)
        except Exception:
            pass
        finally:
            busy.release()
            done.set()

    try:
        threading.Thread(target=run, name="resource-stats-io", daemon=True).start()
    except Exception:
        busy.release()
        return default
    return result[0] if done.wait(_WAIT_SECONDS) else default
