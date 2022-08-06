# Copyright (c) 2022, NVIDIA CORPORATION.
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

from pyhocon import ConfigFactory

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext


def load_config(config_path: str):
    return ConfigFactory.parse_file(config_path)


def get_app_paths(fl_ctx: FLContext) -> (str, str, str):
    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    workspace_dir = workspace.get_root_dir()
    job_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_job_id())
    config_path = f"{job_dir}/config/application.conf"
    return workspace_dir, job_dir, config_path
