# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from builtins import dict as StreamContext

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.widgets.widget import Widget


LOG_STREAM_EVENT_TYPE = "stream_log"


class LogConst(object):
    CLIENT_NAME = "client_name"
    JOB_ID = "job_id"
    LOG_DATA = "log_data"

class LogSender(Widget):
    def __init__(self, event_type=EventType.JOB_COMPLETED, should_report_error_log: bool = True):
        """Sender for analytics data."""
        super().__init__()
        self.event_type = event_type
        self.should_report_error_log = should_report_error_log

    def _stream_error_log_file(self, fl_ctx: FLContext):
        error_log_contents = None
        workspace_root = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        workspace_object = Workspace(root_dir=workspace_root, site_name=client_name)
        error_log_path = workspace_object.get_app_error_log_file_path(job_id=job_id)
        if os.path.exists(error_log_path):
            with open(error_log_path, "r") as f:
                error_log_contents = f.read()
        if error_log_contents:
            FileStreamer.stream_file(
                channel="error_logs",
                topic=LOG_STREAM_EVENT_TYPE,
                stream_ctx={LogConst.CLIENT_NAME: client_name, LogConst.JOB_ID: job_id},
                targets=["server"],
                file_name=error_log_path,
                fl_ctx=fl_ctx,
            )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == self.event_type:
            if self.should_report_error_log:
                t = threading.Thread(target=self._stream_error_log_file, args=(fl_ctx,), daemon=True)
                t.start()
                client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                self.log_info(fl_ctx, f"Started streaming error log file for {client_name} for job {job_id}")

class LogReceiver(Widget):
    def __init__(self):
        """Receives log data."""
        super().__init__()

    def process_log(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Process the streamed log file."""
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        peer_name = peer_ctx.get_identity_name()
        channel = FileStreamer.get_channel(stream_ctx)
        topic = FileStreamer.get_topic(stream_ctx)
        rc = FileStreamer.get_rc(stream_ctx)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Error in streaming log file from {peer_name} on channel {channel} and topic {topic} with rc {rc}")
            return
        file_location = FileStreamer.get_file_location(stream_ctx)
        self.log_info(fl_ctx, f"File location: {file_location}")
        with open(file_location, "r") as f:
            log_contents = f.read()
        client = stream_ctx.get(LogConst.CLIENT_NAME)
        job_id = stream_ctx.get(LogConst.JOB_ID)
        job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
        self.log_info(fl_ctx, f"Saving ERRORLOG from {client} for {job_id}")
        job_manager.set_error_log(job_id, log_contents, client, fl_ctx)


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            FileStreamer.register_stream_processing(fl_ctx, channel="error_logs", topic=LOG_STREAM_EVENT_TYPE, stream_done_cb=self.process_log)
