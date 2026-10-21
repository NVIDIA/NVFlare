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
from tests.timing_utils import WAIT_TIMEOUT


class TestBlobHandlerPoolBlocking:

    def test_blob_handler_should_not_block_pool_worker(self):
        """An admitted application callback must not occupy the streaming pool."""
        callback_started = threading.Event()
        callback_release = threading.Event()
        callback_completed = threading.Event()
        callback_errors = []

        def slow_blob_cb(future, *args, **kwargs):
            try:
                callback_started.set()
                # Only the test may release this gate. A timeout here could
                # free an incorrectly occupied stream worker before the probe.
                callback_release.wait()
            except BaseException as error:
                callback_errors.append(error)
                raise
            finally:
                callback_completed.set()

        handler = BlobHandler(slow_blob_cb)
        future = StreamFuture(stream_id=1, headers={})
        mock_stream = MagicMock()
        mock_stream.get_size.return_value = 0
        mock_stream.read.return_value = b""

        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(handler.handle_blob_cb, future, mock_stream, False)
            try:
                assert callback_started.wait(WAIT_TIMEOUT)
                second = pool.submit(lambda: "stream worker available")
                assert second.result(timeout=WAIT_TIMEOUT) == "stream worker available"
                assert not callback_completed.is_set(), "callback completed before the test released it"
            finally:
                callback_release.set()
                first.result(timeout=WAIT_TIMEOUT)
                assert callback_completed.wait(WAIT_TIMEOUT), "application callback survived test teardown"
                if callback_errors:
                    raise callback_errors[0]
