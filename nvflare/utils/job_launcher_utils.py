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
        e = job_args.get(name)
        if not e:
            continue
        n, v = e
        result += f"{sep}{n} {v}"
        sep = " "
    return result


def get_client_job_args(include_exe_module=True, include_set_options=True):
    result = []
    if include_exe_module:
        result.append(JobProcessArgs.EXE_MODULE)

    result.extend(
        [
            JobProcessArgs.WORKSPACE,
            JobProcessArgs.STARTUP_DIR,
            JobProcessArgs.AUTH_TOKEN,
            JobProcessArgs.TOKEN_SIGNATURE,
            JobProcessArgs.SSID,
            JobProcessArgs.JOB_ID,
            JobProcessArgs.CLIENT_NAME,
            JobProcessArgs.PARENT_URL,
            JobProcessArgs.PARENT_CONN_SEC,
            JobProcessArgs.TARGET,
            JobProcessArgs.SCHEME,
            JobProcessArgs.STARTUP_CONFIG_FILE,
        ]
    )

    if include_set_options:
        result.append(JobProcessArgs.OPTIONS)

    return result


def generate_client_command(fl_ctx) -> str:
    job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    if not job_args:
        raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

    args_str = _job_args_str(job_args, get_client_job_args())
    return f"{sys.executable} {args_str}"


def get_server_job_args(include_exe_module=True, include_set_options=True):
    result = []
    if include_exe_module:
        result.append(JobProcessArgs.EXE_MODULE)

    result.extend(
        [
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
        ]
    )

    if include_set_options:
        result.append(JobProcessArgs.OPTIONS)

    return result


def generate_server_command(fl_ctx) -> str:
    job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    if not job_args:
        raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext!")

    args_str = _job_args_str(job_args, get_server_job_args())
    return f"{sys.executable} {args_str}"


_LAUNCHER_MODE_KEYS = {"process", "docker", "k8s"}


def get_launcher_resource_spec(job_meta, site_name, mode):
    """Extract the resource spec for a site for a specific launcher mode.

    New nested format: resource_spec[site][mode] = {num_of_gpus: ..., shm_size: ..., ...}
    Legacy flat format: resource_spec[site] = {num_of_gpus: ...} — treated as process mode for
    backward compatibility; Docker and K8s modes receive an empty spec.

    Returns a dict for the given mode, or an empty dict if not specified.
    """
    resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
    site_spec = resource_spec.get(site_name) or {}
    if any(k in site_spec for k in _LAUNCHER_MODE_KEYS):
        return site_spec.get(mode, {})
    # Legacy flat format — treat as process only
    return site_spec if mode == "process" else {}


# TODO: remove in follow-up PR once K8s launcher is updated to use get_launcher_resource_spec
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
