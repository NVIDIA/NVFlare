# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.mpm import MainProcessMonitor

STREAM_THREAD_POOL_SIZE = 128
ONE_MB = 1024 * 1024

lock = threading.Lock()
sid_base = int((time.time() + os.getpid()) * 1000000)  # microseconds
stream_count = 0

log = logging.getLogger(__name__)


class CheckedExecutor(ThreadPoolExecutor):
    """This executor ignores task after shutting down"""

    def __init__(self, max_workers=None, thread_name_prefix=""):
        super().__init__(max_workers, thread_name_prefix)
        self.stopped = False

    def shutdown(self, wait=True):
        self.stopped = True
        super().shutdown(wait)

    def submit(self, fn, *args, **kwargs):
        if self.stopped:
            log.debug(f"Call {fn} is ignored after streaming shutting down")
        else:
            super().submit(fn, *args, **kwargs)


stream_thread_pool = CheckedExecutor(STREAM_THREAD_POOL_SIZE, "stm")


def wrap_view(buffer: BytesAlike) -> memoryview:
    if isinstance(buffer, memoryview):
        view = buffer
    else:
        view = memoryview(buffer)

    return view


def gen_stream_id() -> int:
    global lock, stream_count, sid_base
    with lock:
        stream_count += 1
    return sid_base + stream_count


class FastBuffer:
    """A buffer with fast appending"""

    def __init__(self, buf: BytesAlike = None):
        if not buf:
            self.capacity = 1024
        else:
            self.capacity = len(buf)

        self.buffer = bytearray(self.capacity)
        if buf:
            self.buffer[:] = buf
            self.size = len(buf)
        else:
            self.size = 0

    def to_bytes(self) -> BytesAlike:
        """Return bytes-like object.
        Once this method is called, append() may not work any longer, since the buffer may have been exported"""

        if self.capacity == self.size:
            result = self.buffer
        else:
            view = wrap_view(self.buffer)
            result = view[0 : self.size]

        return result

    def append(self, buf: BytesAlike):
        """Fast append by doubling the size of the buffer when it runs out"""

        if not buf:
            return self

        length = len(buf)
        remaining = self.capacity - self.size
        if length > remaining:
            # Expanding the array as least twice the current capacity
            new_cap = max(length + self.size, 2 * self.capacity)
            self.buffer = self.buffer.ljust(new_cap, b"\x00")
            self.capacity = new_cap

        self.buffer[self.size :] = buf
        self.size += length

        return self

    def __len__(self):
        return self.size


def stream_stats_category(fqcn: str, channel: str, topic: str, stream_type: str = "byte"):
    return f"{fqcn}:{stream_type}:{channel}:{topic}"


def stream_shutdown():
    stream_thread_pool.shutdown(wait=True)


MainProcessMonitor.add_cleanup_cb(stream_shutdown)
