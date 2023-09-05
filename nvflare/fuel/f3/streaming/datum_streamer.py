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
from typing import Callable, List, Any

from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer

from nvflare.fuel.f3.message import Message, StreamMessage
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
from nvflare.fuel.f3.streaming.stream_types import StreamFuture

log = logging.getLogger(__name__)

class DatumFuture(StreamFuture):

    def __init__(self, stream_future: StreamFuture, datum_futures: List[StreamFuture]):
        super().__init__(stream_future.get_stream_id(), stream_future.get_headers())
        self.datum_futures = [stream_future]
        if datum_futures:
            self.datum_futures.extend(datum_futures)

        # DatumFuture's size is the total size. If one future has no size, DatumFuture has no size
        total = 0
        for future in self.datum_futures:
            if future.get_size() == 0:
                total = 0
                break
            else:
                total += future.get_size()

        self.set_size(total)

    def get_progress(self) -> int:

        return sum(f.get_progress() for f in self.datum_futures)


    def running(self):
        """Return True if the future is currently executing."""
        with self.lock:
            return not self.waiter.isSet()

    def done(self):
        """Return True of the future was cancelled or finished executing."""
        with self.lock:
            return all(f.done() for f in self.datum_futures)


    def result(self, timeout=None) -> Any:
        """Return the result of the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the result if the future
                isn't done. If None, then there is no limit on the wait time.

        Returns:
            The final result

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
        """

        if not self.waiter.wait(timeout):
            raise TimeoutError(f"Future timed out waiting result after {timeout} seconds")

        if self.error:
            raise self.error

        return self.value

    def exception(self, timeout=None):
        """Return the exception raised by the call that the future represents.

        Args:
            timeout: The number of seconds to wait for the exception if the
                future isn't done. If None, then there is no limit on the wait
                time.

        Returns:
            The exception raised by the call that the future represents or None
            if the call completed without raising.

        Raises:
            CancelledError: If the future was cancelled.
            TimeoutError: If the future didn't finish executing before the given
                timeout.
        """

        if not self.waiter.wait(timeout):
            raise TimeoutError(f"Future timed out waiting exception after {timeout} seconds")

        return self.error

    def set_result(self, value: Any):
        """Sets the return value of work associated with the future."""

        with self.lock:
            if self.error:
                raise StreamError("Invalid state, future already failed")
            self.value = value
            self.waiter.set()

        self._invoke_callbacks()

    def set_exception(self, exception):
        """Sets the result of the future as being the given exception."""
        with self.lock:
            self.error = exception
            self.waiter.set()

        self._invoke_callbacks()

    def _invoke_callbacks(self):
        for callback, args, kwargs in self.done_callbacks:
            try:
                callback(self, args, kwargs)
            except Exception as ex:
                log.error(f"Exception calling callback for {callback}: {ex}")





class DatumStreamer:
    def __init__(self, blob_streamer: BlobStreamer):
        self.blob_streamer = blob_streamer

    def send(self, channel: str, topic: str, target: str, stream_message: StreamMessage,
             progress_cb: Callable) -> StreamFuture:

        if not stream_message.datums:
        blob_stream = BlobStream(stream_message.payload, stream_message.headers)
        return self.byte_streamer.send(channel, topic, target, stream_message.headers)

    def register_blob_callback(self, channel, topic, blob_cb: Callable, *args, **kwargs):
        handler = BlobHandler(blob_cb)
        self.byte_receiver.register_callback(channel, topic, handler.handle_blob_cb, *args, **kwargs)
