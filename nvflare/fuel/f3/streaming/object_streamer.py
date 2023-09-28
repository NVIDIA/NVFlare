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
from typing import Callable, Optional

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.blob_streamer import BlobStreamer
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import ObjectIterator, ObjectStreamFuture, StreamFuture
from nvflare.fuel.f3.streaming.stream_utils import gen_stream_id, stream_thread_pool

log = logging.getLogger(__name__)


class ObjectTxTask:
    def __init__(
        self,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        iterator: ObjectIterator,
        secure: bool,
        optional: bool,
    ):
        self.obj_sid = gen_stream_id()
        self.index = 0
        self.channel = channel
        self.topic = topic
        self.target = target
        self.headers = headers if headers else {}
        self.iterator = iterator
        self.object_future = None
        self.stop = False
        self.secure = secure
        self.optional = optional

    def __str__(self):
        return f"ObjTx[SID:{self.obj_sid}/{self.index} to {self.target} for {self.channel}/{self.topic}]"


class ObjectRxTask:
    def __init__(self, obj_sid: int, channel: str, topic: str, origin: str, headers: dict):
        self.obj_sid = obj_sid
        self.index = 0
        self.channel = channel
        self.topic = topic
        self.origin = origin
        self.headers = headers
        self.object_future: Optional[ObjectStreamFuture] = None

    def __str__(self):
        return f"ObjRx[SID:{self.obj_sid}/{self.index} from {self.origin} for {self.channel}/{self.topic}]"


class ObjectHandler:
    def __init__(self, object_stream_cb: Callable, object_cb: Callable, obj_tasks: dict):
        self.object_stream_cb = object_stream_cb
        self.object_cb = object_cb
        self.obj_tasks = obj_tasks

    def object_done(self, future: StreamFuture, obj_sid: int, index: int, *args, **kwargs):
        blob = future.result()
        self.object_cb(obj_sid, index, Message(future.get_headers(), blob), *args, **kwargs)

    def handle_object(self, future: StreamFuture, *args, **kwargs):
        headers = future.get_headers()
        obj_sid = headers.get(StreamHeaderKey.OBJECT_STREAM_ID, None)

        if obj_sid is None:
            return

        task = self.obj_tasks.get(obj_sid, None)
        if not task:
            # Handle new object stream
            origin = headers.get(MessageHeaderKey.ORIGIN)
            channel = headers.get(StreamHeaderKey.CHANNEL)
            topic = headers.get(StreamHeaderKey.TOPIC)
            task = ObjectRxTask(obj_sid, channel, topic, origin, headers)
            task.object_future = ObjectStreamFuture(obj_sid, headers)

            stream_thread_pool.submit(self.object_stream_cb, task.object_future, *args, **kwargs)

        task.object_future.set_index(task.index)
        task.index += 1
        future.add_done_callback(self.object_done, future, task.obj_sid, task.index)


class ObjectStreamer:
    def __init__(self, blob_streamer: BlobStreamer):
        self.blob_streamer = blob_streamer
        self.obj_tasks = {}

    def stream_objects(
        self,
        channel: str,
        topic: str,
        target: str,
        headers: dict,
        iterator: ObjectIterator,
        secure=False,
        optional=False,
    ) -> ObjectStreamFuture:
        tx_task = ObjectTxTask(channel, topic, target, headers, iterator, secure, optional)
        tx_task.object_future = ObjectStreamFuture(tx_task.obj_sid, headers)
        stream_thread_pool.submit(self._streaming_task, tx_task)

        return tx_task.object_future

    def register_object_callbacks(
        self, channel, topic, object_stream_cb: Callable, object_cb: Callable, *args, **kwargs
    ):
        handler = ObjectHandler(object_stream_cb, object_cb, self.obj_tasks)
        self.blob_streamer.register_blob_callback(channel, topic, handler.handle_object, *args, **kwargs)

    def _streaming_task(self, task: ObjectTxTask):

        for obj in task.iterator:

            task.object_future.set_index(task.index)

            task.headers.update(
                {
                    StreamHeaderKey.OBJECT_STREAM_ID: task.obj_sid,
                    StreamHeaderKey.OBJECT_INDEX: task.index,
                }
            )
            blob_future = self.blob_streamer.send(task.channel, task.topic, task.target, task.headers, obj)

            # Wait till it's done
            bytes_sent = blob_future.result()
            log.debug(f"Stream {task.obj_sid} Object {task.index} is sent ({bytes_sent}")
            task.index += 1

        task.object_future.set_result(task.index)
