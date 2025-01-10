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
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ProcessType, StreamCtxKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import DataTypes
from nvflare.apis.workspace import Workspace
from nvflare.app_common.logging.constants import LOG_STREAM_EVENT_TYPE, Channels
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.widgets.widget import Widget


class ErrorLogSender(Widget):
    def __init__(self, event_type=EventType.JOB_COMPLETED, should_report_error_log: bool = True):
        super().__init__()
        self.event_type = event_type
        self.should_report_error_log = should_report_error_log

    def _stream_log_file(self, fl_ctx: FLContext, log_path: str, log_type: str):
        if os.path.exists(log_path):
            FileStreamer.stream_file(
                channel=Channels.LOG_STREAMING_CHANNEL,
                topic=LOG_STREAM_EVENT_TYPE,
                stream_ctx={
                    StreamCtxKey.CLIENT_NAME: fl_ctx.get_prop(FLContextKey.CLIENT_NAME),
                    StreamCtxKey.JOB_ID: fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID),
                    StreamCtxKey.LOG_TYPE: log_type,
                },
                targets=["server"],
                file_name=log_path,
                fl_ctx=fl_ctx,
            )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == self.event_type:
            if fl_ctx.get_process_type() == ProcessType.CLIENT_PARENT:
                if self.should_report_error_log:
                    workspace_root = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
                    client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
                    job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                    workspace_object = Workspace(root_dir=workspace_root, site_name=client_name)
                    error_log_path = workspace_object.get_app_error_log_file_path(job_id=job_id)

                    if os.path.exists(error_log_path):
                        t = threading.Thread(
                            target=self._stream_log_file,
                            args=(fl_ctx, error_log_path, DataTypes.ERRORLOG.value),
                            daemon=True,
                        )
                        t.start()
                        self.log_info(fl_ctx, f"Started streaming error log file for {client_name} for job {job_id}")
                    else:
                        self.log_info(fl_ctx, f"No error log file found for {client_name} for job {job_id}")
