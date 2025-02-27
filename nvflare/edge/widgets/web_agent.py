#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
import json
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.edge.web.views.eta_views import task_handler
from nvflare.edge.web.web_server import run_server
from nvflare.widgets.widget import Widget


class WebAgent(Widget):
    def __init__(self, port=0, host=""):
        super().__init__()

        self.port = port
        self.host = host
        self.web_thread = None
        self.engine = None

        self.register_event_handler(EventType.SYSTEM_START, self.startup)
        self.register_event_handler(EventType.SYSTEM_END, self.shutdown)

    def run_web_server(self, fl_ctx: FLContext):

        if not self.port:
            # Load json file with port
            workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            mapping_file = workspace.get_file_path_in_root("../lcp_map.json")
            with open(mapping_file, "r") as mapping_file:
                mapping = json.load(mapping_file)
            client_name = fl_ctx.get_identity_name()
            client_config = mapping.get(client_name)
            if client_config is None:
                raise ValueError(f"Client {client_name} not found in {mapping_file}")
            self.host = client_config.get("host")
            self.port = client_config.get("port")

        try:
            run_server(self.host, self.port)
        except Exception as e:
            self.log_error(fl_ctx, f"Web server on port {self.port} stopped due to error: {e}")

    def startup(self, _event_type: str, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        task_handler.set_engine(self.engine)
        self.web_thread = threading.Thread(target=self.run_web_server, args=(fl_ctx,), name="web_server", daemon=True)
        self.web_thread.start()

        self.log_info(fl_ctx, f"Edge web API endpoint is running on port {self.port}")

    def shutdown(self, _event_type: str, fl_ctx: FLContext):
        # Todo: Need to shutdown flask server
        self.log_info(fl_ctx, f"Edge web API endpoint on port {self.port} is shutting down")
