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
from typing import Callable

from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer

from nvflare.fuel.f3.message import Message, StreamMessage
from nvflare.fuel.f3.streaming.byte_receiver import ByteReceiver
from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer
from nvflare.fuel.f3.streaming.stream_types import StreamFuture

log = logging.getLogger(__name__)


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
