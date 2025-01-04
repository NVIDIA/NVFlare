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
from nvflare.apis.job_launcher_spec import JobProcessArgs


def _job_args_str(job_args, arg_names) -> str:
    result = ""
    sep = ""
    for name in arg_names:
        n, v = job_args[name]
        result += f"{sep}{n} {v}"
        sep = " "
    return result


def generate_client_command(job_meta, fl_ctx):
    job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    if not job_args:
        raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

    args_str = _job_args_str(
        job_args,
        [
            JobProcessArgs.EXE_MODULE,
            JobProcessArgs.WORKSPACE,
            JobProcessArgs.STARTUP_DIR,
            JobProcessArgs.AUTH_TOKEN,
            JobProcessArgs.TOKEN_SIGNATURE,
            JobProcessArgs.SSID,
            JobProcessArgs.JOB_ID,
            JobProcessArgs.CLIENT_NAME,
            JobProcessArgs.PARENT_URL,
            JobProcessArgs.TARGET,
            JobProcessArgs.SCHEME,
            JobProcessArgs.STARTUP_CONFIG_FILE,
            JobProcessArgs.OPTIONS,
        ],
    )
    return sys.executable + args_str + " print_conf=True"


def generate_server_command(job_meta, fl_ctx):
    job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    if not job_args:
        raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext!")

    args_str = _job_args_str(
        job_args,
        [
            JobProcessArgs.EXE_MODULE,
            JobProcessArgs.WORKSPACE,
            JobProcessArgs.STARTUP_CONFIG_FILE,
            JobProcessArgs.APP_ROOT,
            JobProcessArgs.JOB_ID,
            JobProcessArgs.TOKEN_SIGNATURE,
            JobProcessArgs.PARENT_URL,
            JobProcessArgs.ROOT_URL,
            JobProcessArgs.SERVICE_HOST,
            JobProcessArgs.SERVICE_PORT,
            JobProcessArgs.SSID,
            JobProcessArgs.HA_MODE,
            JobProcessArgs.OPTIONS,
        ],
    )

    return sys.executable + args_str + f" restore_snapshot={job_args[JobProcessArgs.RESTORE_SNAPSHOT]} print_conf=True"


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
