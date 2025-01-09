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
import copy
import os
import sys

from nvflare.apis.fl_constant import FLContextKey, JobConstants, SystemVarName
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace


def generate_client_command(job_meta, fl_ctx):
    workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    args = fl_ctx.get_prop(FLContextKey.ARGS)
    client = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
    job_id = job_meta.get(JobConstants.JOB_ID)
    server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
    if not server_config:
        raise RuntimeError(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
    service = server_config[0].get("service", {})
    if not isinstance(service, dict):
        raise RuntimeError(f"expect server config data to be dict but got {type(service)}")
    command_options = ""
    for t in args.set:
        command_options += " " + t
    command = (
        f"{sys.executable} -m nvflare.private.fed.app.client.worker_process -m "
        + args.workspace
        + " -w "
        + (workspace_obj.get_startup_kit_dir())
        + " -t "
        + client.token
        + " -d "
        + client.ssid
        + " -n "
        + job_id
        + " -c "
        + client.client_name
        + " -p "
        + str(client.cell.get_internal_listener_url())
        + " -g "
        + service.get("target")
        + " -scheme "
        + service.get("scheme", "grpc")
        + " -s fed_client.json "
        " --set" + command_options + " print_conf=True"
    )
    return command


def generate_server_command(job_meta, fl_ctx):
    workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    args = fl_ctx.get_prop(FLContextKey.ARGS)
    server = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
    job_id = job_meta.get(JobConstants.JOB_ID)
    restore_snapshot = fl_ctx.get_prop(FLContextKey.SNAPSHOT, False)
    app_root = workspace_obj.get_app_dir(job_id)
    cell = server.cell
    server_state = server.server_state
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
    return command


def extract_job_image(job_meta, site_name):
    deploy_map = job_meta.get(JobMetaKey.DEPLOY_MAP, {})
    for _, participants in deploy_map.items():
        for item in participants:
            if isinstance(item, dict):
                sites = item.get(JobConstants.SITES)
                if site_name in sites:
                    return item.get(JobConstants.JOB_IMAGE)
    return None


def add_custom_dir_to_path(app_custom_folder, new_env):
    """Util method to add app_custom_folder into the sys.path and carry into the child process."""
    sys_path = copy.copy(sys.path)
    sys_path.append(app_custom_folder)
    new_env[SystemVarName.PYTHONPATH] = os.pathsep.join(sys_path)
