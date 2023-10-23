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
import os
from typing import Callable

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_TYPE_BYTE, ByteStreamer
from nvflare.fuel.f3.streaming.file_streamer import FileStreamer
from nvflare.fuel.f3.streaming.object_streamer import ObjectStreamer
from nvflare.fuel.f3.streaming.stream_types import ObjectIterator, ObjectStreamFuture, Stream, StreamError, StreamFuture


class StreamCell:
    def __init__(self, cell: CoreCell):
        self.cell = cell
        self.byte_streamer = ByteStreamer(cell)
        self.byte_receiver = ByteReceiver(cell)
        self.blob_streamer = BlobStreamer(self.byte_streamer, self.byte_receiver)
        self.file_streamer = FileStreamer(self.byte_streamer, self.byte_receiver)
        self.object_streamer = ObjectStreamer(self.blob_streamer)

    @staticmethod
    def get_chunk_size():
        """Get the default chunk size used by StreamCell
        Byte stream are broken into chunks of this size before sending over Cellnet
        """
        return ByteStreamer.get_chunk_size()

    def send_stream(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> StreamFuture:
        """
        Send a byte-stream over a channel/topic asynchronously. The streaming is performed in a different thread.
        The streamer will read from stream and send the data in chunks till the stream reaches EOF.

        Args:
            channel: channel for the stream
            topic: topic for the stream
            target: destination cell FQCN
            message: The payload is the stream to send
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns: StreamFuture that can be used to check status/progress, or register callbacks.
            The future result is the number of bytes sent

        """

        if not isinstance(message.payload, Stream):
            raise StreamError(f"Message payload is not a stream: {type(message.payload)}")

        return self.byte_streamer.send(
            channel, topic, target, message.headers, message.payload, STREAM_TYPE_BYTE, secure, optional
        )

    def register_stream_cb(self, channel: str, topic: str, stream_cb: Callable, *args, **kwargs):
        """
        Register a callback for reading stream. The stream_cb must have the following signature,
            stream_cb(future: StreamFuture, stream: Stream, resume: bool, *args, **kwargs) -> int
                future: The future represents the ongoing streaming. It's done when streaming is complete.
                stream: The stream to read the receiving data from
                resume: True if this is a restarted stream
                It returns the offset to resume from if this is a restarted stream

        The resume_cb returns the offset to resume from:
            resume_cb(stream_id: str, *args, **kwargs) -> int

        If None, the stream is not resumable.

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
        """
        Send a BLOB (Binary Large Object) to the target. The payload of message is the BLOB. The BLOB must fit in
        memory on the receiving end.

        Args:
            channel: channel for the message
            topic: topic of the message
            target: destination cell IDs
            message: the headers and the blob as payload
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns: StreamFuture that can be used to check status/progress and get result
            The future result is the total number of bytes sent

        """

        if message.payload is None:
            message.payload = bytes(0)

        if not isinstance(message.payload, (bytes, bytearray, memoryview, list)):
            raise StreamError(f"Message payload is not a byte array: {type(message.payload)}")

        return self.blob_streamer.send(channel, topic, target, message, secure, optional)

    def register_blob_cb(self, channel: str, topic: str, blob_cb, *args, **kwargs):
        """
        Register a callback for receiving the blob. This callback is invoked when the whole
        blob is received. If streaming fails, the streamer will try again. The failed streaming
        is ignored.

        The callback must have the following signature,
            blob_cb(future: StreamFuture, *args, **kwargs)

        The future's result is the final BLOB received

        Args:
            channel: the channel of the request
            topic: topic of the request
            blob_cb: The callback to handle the stream
        """
        self.blob_streamer.register_blob_callback(channel, topic, blob_cb, *args, **kwargs)

    def send_file(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> StreamFuture:
        """
        Send a file to target using stream API.

        Args:
            channel: channel for the message
            topic: topic for the message
            target: destination cell FQCN
            message: the headers and the full path of the file to be sent as payload
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns: StreamFuture that can be used to check status/progress and get the total bytes sent

        """
        if not isinstance(message.payload, str):
            raise StreamError(f"Message payload is not a file name: {type(message.payload)}")

        file_name = message.payload
        if not os.path.isfile(file_name) or not os.access(file_name, os.R_OK):
            raise StreamError(f"File {file_name} doesn't exist or isn't readable")

        return self.file_streamer.send(channel, topic, target, message, secure, optional)

    def register_file_cb(self, channel: str, topic: str, file_cb, *args, **kwargs):
        """
        Register callbacks for file receiving. The callbacks must have the following signatures,
            file_cb(future: StreamFuture, file_name: str, *args, **kwargs) -> str
                The future represents the file receiving task and the result is the final file path
                It returns the full path where the file will be written to

        Args:
            channel: the channel of the request
            topic: topic of the request
            file_cb: This CB is called when file transfer starts
        """
        self.file_streamer.register_file_callback(channel, topic, file_cb, *args, **kwargs)

    def send_objects(
        self, channel: str, topic: str, target: str, message: Message, secure=False, optional=False
    ) -> ObjectStreamFuture:
        """
        Send a list of objects to the destination. Each object is sent as BLOB, so it must fit in memory

        Args:
            channel: channel for the message
            topic: topic of the message
            target: destination cell IDs
            message: Headers and the payload which is an iterator that provides next object
            secure: Send the message with end-end encryption if True
            optional: Optional message, error maybe suppressed

        Returns: ObjectStreamFuture that can be used to check status/progress, or register callbacks
        """
        if not isinstance(message.payload, ObjectIterator):
            raise StreamError(f"Message payload is not an object iterator: {type(message.payload)}")

        return self.object_streamer.stream_objects(
            channel, topic, target, message.headers, message.payload, secure, optional
        )

    def register_objects_cb(
        self, channel: str, topic: str, object_stream_cb: Callable, object_cb: Callable, *args, **kwargs
    ):
        """
        Register callback for receiving the object. The callback signature is,
            objects_stream_cb(future: ObjectStreamFuture, resume: bool, *args, **kwargs) -> int
                future: It represents the streaming of all objects. An object CB can be registered with the future
                to receive each object.
                resume: True if this is a restarted stream
                This CB returns the index to restart if this is a restarted stream

            object_cb(obj_sid: str, index: int, message: Message, *args, ** kwargs)
                obj_sid: Object Stream ID
                index: The index of the object
                message: The header and payload is the object

            resume_cb(stream_id: str, *args, **kwargs) -> int
        is received. The index starts from 0. The callback must have the following signature,
            objects_cb(future: ObjectStreamFuture, index: int, object: Any, headers: Optional[dict], *args, **kwargs)
            resume_cb(stream_id: str, *args, **kwargs) -> int

        Args:
            channel: the channel of the request
            topic: topic of the request
            object_stream_cb: The callback when an object stream is started
            object_cb: The callback is invoked when each object is received
        """
        self.object_streamer.register_object_callbacks(channel, topic, object_stream_cb, object_cb, args, kwargs)
