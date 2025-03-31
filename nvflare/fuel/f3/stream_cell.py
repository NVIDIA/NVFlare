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
from typing import Callable

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_TYPE_BYTE, ByteStreamer
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError, StreamFuture


class StreamCell:
    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.byte_streamer = ByteStreamer(cell)
        self.byte_receiver = ByteReceiver(cell)
        self.blob_streamer = BlobStreamer(self.byte_streamer, self.byte_receiver)

    def get_chunk_size(self):
        """Gets the default chunk size used by StreamCell.

        Byte stream are broken into chunks of this size before sending over Cellnet
        """
        return self.byte_streamer.get_chunk_size()

    def send_stream(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> StreamFuture:
        """Sends a byte-stream over a channel/topic asynchronously. The streaming is performed in a different thread.

        The streamer will read from stream and send the data in chunks till the stream reaches EOF.

        Args:
            channel: channel for the stream
            topic: topic for the stream
            target: destination cell FQCN
            message: The payload is the stream to send
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns:
            A StreamFuture that can be used to check status/progress, or register callbacks.
            The future result is the number of bytes sent

        """

        if not isinstance(message.payload, Stream):
            raise StreamError(f"Message payload is not a stream: {type(message.payload)}")

        return self.byte_streamer.send(
            channel, topic, target, message.headers, message.payload, STREAM_TYPE_BYTE, secure, optional
        )

    def register_stream_cb(self, channel: str, topic: str, stream_cb: Callable, *args, **kwargs):
        """Registers a callback for reading stream.

        The stream_cb must have the following signature:

        .. code-block: python

            stream_cb(future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int
                future: The future represents the ongoing streaming. It's done when streaming is complete.
                stream: The stream to read the receiving data from
                resume: True if this is a restarted stream
                    It returns the offset to resume from if this is a restarted stream

        Args:
            channel: the channel of the request
            topic: topic of the request
            stream_cb: The callback to handle the stream. This is called when a stream is started. It also
                provides restart offset for restarted streams. This CB is invoked in a dedicated thread,
                and it can block
            *args: positional args to be passed to the callbacks
            **kwargs: keyword args to be passed to the callbacks

        """
        self.byte_receiver.register_callback(channel, topic, stream_cb, *args, **kwargs)

    def send_blob(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> StreamFuture:
        """Sends a BLOB (Binary Large Object) to the target.

        The payload of message is the BLOB. The BLOB must fit in
        memory on the receiving end.

        Args:
            channel: channel for the message
            topic: topic of the message
            target: destination cell IDs
            message: the headers and the blob as payload
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns:
            StreamFuture that can be used to check status/progress and get result
            The future result is the total number of bytes sent

        """

        if message.payload is None:
            message.payload = bytes(0)

        if not isinstance(message.payload, (bytes, bytearray, memoryview, list)):
            raise StreamError(f"Message payload is not a byte array: {type(message.payload)}")

        return self.blob_streamer.send(channel, topic, target, message, secure, optional)

    def register_blob_cb(self, channel: str, topic: str, blob_cb, *args, **kwargs):
        """Registers a callback for receiving the blob.

        This callback is invoked when the whole blob is received.
        If streaming fails, the streamer will try again. The failed streaming
        is ignored.

        The callback must have the following signature:

        .. code-block: python

            blob_cb(future: StreamFuture, *args, **kwargs)

        The future's result is the final BLOB received

        Args:
            channel: the channel of the request
            topic: topic of the request
            blob_cb: The callback to handle the stream
        """
        self.blob_streamer.register_blob_callback(channel, topic, blob_cb, *args, **kwargs)
