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

"""
Tests that BlobHandler.handle_blob_cb does not block the pool worker thread.

Root cause: stream_thread_pool was shared for blob reads AND application callbacks.
BlobHandler.handle_blob_cb ran blob_cb synchronously, blocking the worker for
the entire duration of produce() + send_blob(). Under load, this starved the
pool and delayed reply processing past timeout.

Fix: blob_cb is now dispatched to a separate callback_thread_pool, so the
pool worker is freed immediately after submitting the callback.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from nvflare.fuel.f3.streaming.blob_streamer import BlobHandler
from nvflare.fuel.f3.streaming.stream_types import StreamFuture


class TestBlobHandlerPoolBlocking:

    def test_blob_handler_should_not_block_pool_worker(self):
        """Pool worker should be free to handle other tasks while blob_cb executes."""
        callback_started = threading.Event()
        callback_release = threading.Event()

        def slow_blob_cb(future, *args, **kwargs):
            callback_started.set()
            callback_release.wait()  # simulates expensive produce + send_blob

        handler = BlobHandler(slow_blob_cb)
        mock_future = MagicMock(spec=StreamFuture)
        mock_future.headers = {}
        mock_stream = MagicMock()
        mock_stream.get_size.return_value = 0
        mock_stream.read.return_value = b""

        pool = ThreadPoolExecutor(max_workers=1)

        def wrapper():
            handler.handle_blob_cb(mock_future, mock_stream, False)

        pool.submit(wrapper)
        assert callback_started.wait(timeout=2.0)

        # With correct behavior, the pool worker should be free to run other tasks
        # even while blob_cb is still executing.
        second_ran = threading.Event()
        pool.submit(lambda: second_ran.set())
        ran = second_ran.wait(timeout=0.5)

        callback_release.set()
        pool.shutdown(wait=True)

        assert ran, (
            "BUG: pool worker is blocked by synchronous blob_cb execution. "
            "handle_blob_cb should not hold the worker while the callback runs."
        )
