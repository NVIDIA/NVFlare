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

from http import client
import os
from threading import Lock
import time
from typing import List, Optional
from builtins import dict as StreamContext

from flask.cli import F

from nvflare.apis.dxo import DXO, DataKind
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
        """Sender for analytics data.

        Args:
            event_type (str): event type to fire (defaults to "stream_log").
        """
        super().__init__()
        self.engine = None
        self.event_type = event_type
        self.should_report_error_log = should_report_error_log

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type != EventType.AFTER_CLIENT_HEARTBEAT and event_type != EventType.BEFORE_CLIENT_HEARTBEAT:
            self.log_error(fl_ctx, "LogSender GOT EVENT: {}".format(event_type), fire_event=False)
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.engine = fl_ctx.get_engine()
            self.log_error(fl_ctx, "LogSender GOT initialized: {}".format(event_type))
            print("LogSender initialized")
            print("*" * 30)
            print(f"Should report error log: {self.should_report_error_log}")
        elif event_type == self.event_type:
            self.log_error(fl_ctx, "LogSender GOT event_type: {}".format(event_type))
            if self.should_report_error_log:
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

    def close(self):
        """Close resources."""
        if self.engine:
            self.engine = None


class LogReceiver(Widget):
    def __init__(self, events: Optional[List[str]] = None):
        """Receives log data.

        Args:
            events (optional, List[str]): A list of event that this receiver will handle.
        """
        super().__init__()
        self.engine = None
        if events is None:
            events = [LOG_STREAM_EVENT_TYPE, f"fed.{LOG_STREAM_EVENT_TYPE}"]
        self.events = events
        self._save_lock = Lock()

    def process_log(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Process the streamed log file."""
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        peer_name = peer_ctx.get_identity_name()
        channel = FileStreamer.get_channel(stream_ctx)
        topic = FileStreamer.get_topic(stream_ctx)
        rc = FileStreamer.get_rc(stream_ctx)
        self.log_info(fl_ctx, f"Received log file from {peer_name} on channel {channel} and topic {topic} with rc {rc}")
        self.log_info(fl_ctx, f"Stream context {stream_ctx}")
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Error in streaming log file from {peer_name} on channel {channel} and topic {topic} with rc {rc}")
            return
        file_location = FileStreamer.get_file_location(stream_ctx)
        self.log_info(fl_ctx, f"File location: {file_location}")
        # Check file size before reading
        file_size = os.path.getsize(file_location)
        self.log_info(fl_ctx, f"File size: {file_size} bytes")
        with open(file_location, "r") as f:
            log_contents = f.read()
        self.log_info(fl_ctx, "GOT log contents!!!")
        self.log_info(fl_ctx, log_contents)
        client = stream_ctx.get(LogConst.CLIENT_NAME)
        job_id = stream_ctx.get(LogConst.JOB_ID)
        job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
        self.log_info(fl_ctx, f"TRYING TO SAVE ERROR LOG from {client} for {job_id}")
        job_manager.set_error_log(job_id, log_contents, client, fl_ctx)


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type != EventType.CLIENT_HEARTBEAT_RECEIVED and event_type != EventType.CLIENT_HEARTBEAT_PROCESSED:
            self.log_error(fl_ctx, "LogReceiver GOT EVENT: {}".format(event_type), fire_event=False)
        if event_type == EventType.SYSTEM_START:
            FileStreamer.register_stream_processing(fl_ctx, channel="error_logs", topic=LOG_STREAM_EVENT_TYPE, stream_done_cb=self.process_log)
