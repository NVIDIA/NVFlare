# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import random

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.file_retriever import FileRetriever
from nvflare.app_common.streamers.file_streamer import FileStreamer, StreamContext

from .defs import STREAM_CHANNEL, TOPIC_ECHO_FILE, TOPIC_INITIAL_FILE


class AppCmdExecutor(Executor):
    def __init__(self, file_retriever_id=None):
        Executor.__init__(self)
        self.file_retriever_id = file_retriever_id
        self.file_retriever = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            engine.register_aux_message_handler(
                topic="echo",
                message_handle_func=self._handle_echo,
            )
            FileStreamer.register_stream_processing(
                fl_ctx, STREAM_CHANNEL, TOPIC_INITIAL_FILE, stream_status_cb=self._file_received, file_type="initial"
            )
            FileStreamer.register_stream_processing(
                fl_ctx, STREAM_CHANNEL, TOPIC_ECHO_FILE, stream_status_cb=self._file_received, file_type="echo"
            )

            if self.file_retriever_id:
                c = engine.get_component(self.file_retriever_id)
                if not isinstance(c, FileRetriever):
                    self.system_panic(
                        f"invalid file_retriever {self.file_retriever_id}: expect FileRetriever but got {type(c)}",
                        fl_ctx,
                    )
                    return
                self.file_retriever = c

    def _file_received(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
        file_type: str,
    ):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        peer_name = peer_ctx.get_identity_name()
        channel = FileStreamer.get_channel(stream_ctx)
        topic = FileStreamer.get_topic(stream_ctx)
        rc = FileStreamer.get_rc(stream_ctx)
        self.log_info(fl_ctx, f"file received from {peer_name}: {stream_ctx=} {file_type=} {channel=} {topic=} {rc=}")
        file_location = FileStreamer.get_file_location(stream_ctx)
        if file_type == "initial":
            # send the file back to everyone
            self.log_info(fl_ctx, f"echo file to all: {file_location}")
            streamed = FileStreamer.stream_file(
                channel=STREAM_CHANNEL,
                topic=TOPIC_ECHO_FILE,
                targets="@ALL",
                file_name=file_location,
                fl_ctx=fl_ctx,
                stream_ctx={"file_type": file_type},
            )
            self.log_info(fl_ctx, f"streamed echo file to all sites: {streamed}")

    def _handle_echo(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        data = request.get("data")
        s = Shareable()
        s["data"] = data
        return s

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"got task {task_name}: {shareable}")
        if task_name == "hello":
            data = shareable.get("data")
            s = Shareable()
            s["data"] = data
            return s
        elif task_name == "avg":
            data = shareable.get("data")

            self.log_info(fl_ctx, f"got avg request: {shareable}")

            start = data.get("start", 0)
            end = data.get("end", 0)
            v = random.randint(start, end)
            result = Shareable()
            result["data"] = v
            return result
        elif task_name == "rtr_file":
            file_name = shareable.get("file_name")
            if not file_name:
                self.log_error(fl_ctx, "missing file name in request")
                return make_reply(ReturnCode.BAD_TASK_DATA)
            if not self.file_retriever:
                self.log_error(fl_ctx, "no file retriever")
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

            assert isinstance(self.file_retriever, FileRetriever)
            rc, location = self.file_retriever.retrieve_file(
                from_site="server",
                fl_ctx=fl_ctx,
                timeout=10.0,
                file_name=file_name,
            )
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"failed to retrieve file {file_name}: {rc}")
                return make_reply(rc)
            self.log_info(fl_ctx, f"received file {location}")
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"got unknown task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
