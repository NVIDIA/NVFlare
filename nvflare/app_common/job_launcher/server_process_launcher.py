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
import sys

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.workspace import Workspace
from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.private.fed.utils.fed_utils import add_custom_dir_to_path


class ServerProcessJobLauncher(ProcessJobLauncher):
    def get_command(self, launch_data, fl_ctx) -> (str, dict):
        new_env = os.environ.copy()

        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        server = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
        job_id = launch_data.get(JobConstants.JOB_ID)
        restore_snapshot = fl_ctx.get_prop(FLContextKey.SNAPSHOT, False)

        app_root = workspace_obj.get_app_dir(job_id)
        cell = server.cell
        server_state = server.server_state

        app_custom_folder = workspace_obj.get_app_custom_dir(job_id)
        if app_custom_folder != "":
            add_custom_dir_to_path(app_custom_folder, new_env)

        command_options = ""
        for t in args.set:
            command_options += " " + t

        command = (
            sys.executable
            + " -m nvflare.private.fed.app.server.runner_process -m "
            + args.workspace
            + " -s fed_server.json -r "
            + app_root
            + " -n "
            + str(job_id)
            + " -p "
            + str(cell.get_internal_listener_url())
            + " -u "
            + str(cell.get_root_url_for_child())
            + " --host "
            + str(server_state.host)
            + " --port "
            + str(server_state.service_port)
            + " --ssid "
            + str(server_state.ssid)
            + " --ha_mode "
            + str(server.ha_mode)
            + " --set"
            + command_options
            + " print_conf=True restore_snapshot="
            + str(restore_snapshot)
        )

        return command, new_env
