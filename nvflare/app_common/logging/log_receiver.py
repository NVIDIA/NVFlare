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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode, StreamCtxKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.logging.constants import Channels
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.widgets.widget import Widget


class LogReceiver(Widget):
    def __init__(self):
        """Receives log data.

        If adding additional log types, make sure nvflare.apis.storage.ComponentPrefixes has
        the corresponding log type.
        """
        super().__init__()

    def process_log(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Process the streamed log file."""
        rc = FileStreamer.get_rc(stream_ctx)
        if rc != ReturnCode.OK:
            peer_ctx = fl_ctx.get_peer_context()
            peer_name = peer_ctx.get_identity_name()
            channel = FileStreamer.get_channel(stream_ctx)
            topic = FileStreamer.get_topic(stream_ctx)
            self.log_error(
                fl_ctx,
                f"Error in streaming log file from {peer_name=} and {topic=}/{channel=} with {rc=}",
            )
            return
        file_location = FileStreamer.get_file_location(stream_ctx)
        client = stream_ctx.get(StreamCtxKey.CLIENT_NAME)
        job_id = stream_ctx.get(StreamCtxKey.JOB_ID)
        job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
        log_type = stream_ctx.get(StreamCtxKey.LOG_TYPE)
        self.log_info(fl_ctx, f"Saving {log_type} from {client} for {job_id}")
        job_manager.set_client_data(job_id, file_location, client, log_type, fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            FileStreamer.register_stream_processing(
                fl_ctx,
                channel=Channels.LOG_STREAMING_CHANNEL,
                topic="*",
                stream_done_cb=self.process_log,
            )
