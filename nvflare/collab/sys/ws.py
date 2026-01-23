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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace as NVFWorkspace
from nvflare.collab.api.workspace import Workspace


class FlareWorkspace(Workspace):

    def __init__(self, fl_ctx: FLContext):
        super().__init__()
        ws_obj = fl_ctx.get_workspace()
        if not isinstance(ws_obj, NVFWorkspace):
            raise RuntimeError(f"the ws_obj must be NVFWorkspace but got {type(ws_obj)}")
        self.flare_ws = ws_obj
        self.job_id = fl_ctx.get_job_id()

    def get_root_dir(self) -> str:
        return self.flare_ws.get_root_dir()

    def get_work_dir(self) -> str:
        return self.flare_ws.get_run_dir(self.job_id)

    def get_experiment_dir(self) -> str:
        return self.get_work_dir()
