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
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as futures_thread

from nvflare.fuel.f3.connection import BytesAlike
from nvflare.fuel.f3.mpm import MainProcessMonitor

STREAM_THREAD_POOL_SIZE = 128
CALLBACK_THREAD_POOL_SIZE = 64
DOWNLOAD_REQUEST_THREAD_POOL_SIZE = 64
ONE_MB = 1024 * 1024
MILLION = 1000000

lock = threading.Lock()
sid_base = int(time.time() * MILLION) + random.randint(0, MILLION)  # microseconds + random
stream_count = 0

log = logging.getLogger(__name__)


class CheckedExecutor(ThreadPoolExecutor):
    """This executor ignores task after shutting down"""

    def __init__(self, max_workers=None, thread_name_prefix=""):
        super().__init__(max_workers, thread_name_prefix)
        self.stopped = False
        # ThreadPoolExecutor protects its own shutdown flag, but our public
        # ``stopped`` check used to happen before entering that protection. A
        # concurrent shutdown could therefore land between the check and
        # ``super().submit()`` and raise "cannot schedule new futures after
        # shutdown" on an F3 worker during process teardown.
        self._lifecycle_lock = threading.Lock()

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Mark both our wrapper and the base executor stopped while submitters
        # are excluded. Do not hold this lock while waiting for workers: an
        # already-admitted task may itself make a best-effort nested submission.
        with self._lifecycle_lock:
            if not self.stopped:
                self.stopped = True
                super().shutdown(wait=False, cancel_futures=cancel_futures)
        if wait:
            super().shutdown(wait=True, cancel_futures=cancel_futures)

    def submit(self, fn, *args, **kwargs):
        with self._lifecycle_lock:
            if self.stopped:
                log.debug(f"Call {fn} is ignored after streaming shutting down")
                return None
            try:
                return super().submit(fn, *args, **kwargs)
            except RuntimeError:
                # Keep the wrapper fail-closed even if a caller reached the base
                # executor's shutdown path directly, or CPython interpreter
                # teardown set the module-global executor shutdown state without
                # entering this override.
                if self._shutdown or futures_thread._shutdown:
                    self.stopped = True
                    log.debug(f"Call {fn} is ignored after the executor shut down")
                    return None
                raise


stream_thread_pool = CheckedExecutor(STREAM_THREAD_POOL_SIZE, "stm")
callback_thread_pool = CheckedExecutor(CALLBACK_THREAD_POOL_SIZE, "stm_cb")
download_request_thread_pool = CheckedExecutor(DOWNLOAD_REQUEST_THREAD_POOL_SIZE, "stm_dl")


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
    download_request_thread_pool.shutdown(wait=True)
    callback_thread_pool.shutdown(wait=True)
    stream_thread_pool.shutdown(wait=True)


MainProcessMonitor.add_cleanup_cb(stream_shutdown)
