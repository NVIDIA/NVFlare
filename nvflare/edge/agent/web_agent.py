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
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.edge.web.web_server import run_server


class WebAgent(FLComponent):
    def __init__(self, port, job_handler_id=None):
        FLComponent.__init__(self)

        self.port = port
        self.job_handler_id = job_handler_id
        self.web_thread = None

    def run_web_server(self, fl_ctx: FLContext):
        try:

            run_server(self.port)

        except Exception as e:
            self.log_error(fl_ctx, f"Web server on port {self.port} stopped due to error: {e}")

    def startup(self, fl_ctx: FLContext):

        self.web_thread = threading.Thread(target=self.run_web_server, args=(fl_ctx,), name="web_server", daemon=True)
        self.web_thread.start()

        self.log_info(fl_ctx, f"Edge web API endpoint is running on port {self.port}")

    def shutdown(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Edge web API endpoint on port {self.port} is shutting down")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.startup(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.shutdown(fl_ctx)
