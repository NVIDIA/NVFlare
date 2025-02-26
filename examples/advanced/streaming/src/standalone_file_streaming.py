# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
from threading import Thread

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.streamers.file_streamer import FileStreamer

CHANNEL = "_test_channel"
TOPIC = "_test_topic"
SIZE = 100 * 1024 * 1024  # 100 MB


class FileSender(FLComponent):
    def __init__(self):
        super().__init__()
        self.seq = 0
        self.aborted = False
        self.file_name = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.log_info(fl_ctx, "FileSender is started")
            Thread(target=self._sending_file, args=(fl_ctx,), daemon=True).start()
        elif event_type == EventType.ABORT_TASK:
            self.log_info(fl_ctx, "Sender is aborted")
            self.aborted = True

    def _sending_file(self, fl_ctx):

        # Create a temp file to send
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            buf = bytearray(SIZE)
            for i in range(len(buf)):
                buf[i] = i % 256

            tmp.write(buf)
        finally:
            tmp.close()

        self.file_name = tmp.name

        rc, result = FileStreamer.stream_file(
            targets=["server"],
            stream_ctx={},
            channel=CHANNEL,
            topic=TOPIC,
            file_name=self.file_name,
            fl_ctx=fl_ctx,
            optional=False,
            secure=False,
        )

        self.log_info(fl_ctx, f"Sending finished with RC: {rc}")
        os.remove(self.file_name)


class FileReceiver(FLComponent):
    def __init__(self):
        super().__init__()
        self.done = False

    def is_done(self):
        return self.done

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._receive_file(fl_ctx)
            self.log_info(fl_ctx, "FileReceiver is started")

    def _receive_file(self, fl_ctx):
        FileStreamer.register_stream_processing(
            fl_ctx=fl_ctx,
            channel=CHANNEL,
            topic=TOPIC,
            stream_done_cb=self._done_cb,
        )

    def _done_cb(self, stream_ctx: dict, fl_ctx: FLContext):
        self.log_info(fl_ctx, "File streaming is done")
        self.done = True

        file_name = FileStreamer.get_file_location(stream_ctx)
        file_size = FileStreamer.get_file_size(stream_ctx)
        size = os.path.getsize(file_name)

        if size == file_size:
            self.log_info(fl_ctx, f"File {file_name} has correct size {size} bytes")
        else:
            self.log_error(fl_ctx, f"File {file_name} sizes mismatch {size} <> {file_size} bytes")

        os.remove(file_name)
