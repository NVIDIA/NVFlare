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
import os
import time
from threading import Thread

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.streamers.file_streamer import FileStreamer

CHANNEL = "_test_channel"
TOPIC = "_test_topic"
TIMESTAMP = "_timestamp"
FILE_NAME = "_filename"
SITE_NAME = "_site_name"


class FileSender(FLComponent):

    def __init__(self, file_name: str, timeout=None):
        super().__init__()
        self.seq = 0
        self.aborted = False
        self.file_name = file_name
        self.timeout = timeout

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.log_info(fl_ctx, "FileSender is started")
            Thread(target=self._sending_file, args=(fl_ctx,), daemon=True).start()
        elif event_type == EventType.ABORT_TASK:
            self.log_info(fl_ctx, "Sender is aborted")
            self.aborted = True

    def _sending_file(self, fl_ctx):
        try:
            self.log_info(fl_ctx, f"Sending file {self.file_name}")
            context = {
                TIMESTAMP: time.time(),
                FILE_NAME: os.path.basename(self.file_name),
                SITE_NAME: fl_ctx.get_identity_name(),
            }
            rc, result = FileStreamer.stream_file(
                targets=["server"],
                stream_ctx=context,
                channel=CHANNEL,
                topic=TOPIC,
                file_name=self.file_name,
                fl_ctx=fl_ctx,
                chunk_timeout=self.timeout,
                optional=False,
                secure=False,
            )

            self.log_info(fl_ctx, f"Sending finished with RC: {rc}")
        except Exception as e:
            self.log_error(fl_ctx, f"Error sending file: {e}")


class FileReceiver(FLComponent):

    def __init__(self, output_folder: str):
        super().__init__()
        self.output_folder = output_folder
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
            dest_dir=self.output_folder,
            stream_done_cb=self._done_cb,
        )

    def _done_cb(self, stream_ctx: dict, fl_ctx: FLContext):
        rc = stream_ctx[StreamContextKey.RC]
        self.log_info(fl_ctx, f"File streaming is done with RC: {rc}")
        self.done = True

        basename = stream_ctx.get(FILE_NAME, "No Name")
        site_name = stream_ctx.get(SITE_NAME, "unknown")
        file_name = os.path.join(self.output_folder, site_name + "_" + basename)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"File {file_name} receiving failed with RC: {rc})")
            return

        file_location = FileStreamer.get_file_location(stream_ctx)
        file_size = FileStreamer.get_file_size(stream_ctx)
        size = os.path.getsize(file_location)
        if size != file_size:
            self.log_error(fl_ctx, f"File {file_location} sizes mismatch {size} <> {file_size} bytes")
            return

        if os.path.exists(file_name):
            self.log_info(fl_ctx, f"Existing file {file_name} is removed")
            os.remove(file_name)

        os.rename(file_location, file_name)

        start_time = stream_ctx.get(TIMESTAMP)
        duration = time.time() - start_time
        self.log_info(fl_ctx, f"File {file_name} with {size} bytes received in {duration:.3f} seconds")
