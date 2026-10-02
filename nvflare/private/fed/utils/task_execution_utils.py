# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Read-only task-runtime capability checks; never build application components."""

import copy
import io
import os
from zipfile import ZipFile

from nvflare.apis.fl_constant import SystemVarName
from nvflare.apis.task_execution import ExecutionLifetime
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.fuel.utils.json_scanner import JsonScanner
from nvflare.fuel.utils.wfconf import resolve_var_refs

TASK_RUNTIME_CAPABILITY = "task_execution_process_v1"
RUNTIME_CAPABILITIES = "runtime_capabilities"


def execution_lifetime(config, variables=None):
    value = {"execution_lifetime": copy.deepcopy(config.get("execution_lifetime", ExecutionLifetime.JOB))}
    all_vars = copy.deepcopy(config)
    all_vars.update(variables or {})
    resolve_var_refs(JsonScanner(value), all_vars)
    return ExecutionLifetime.validate(value["execution_lifetime"])


def client_app_uses_task_runtime(workspace, job_id, variables):
    variables = dict(variables)
    variables.update(
        {
            SystemVarName.JOB_ID: job_id,
            SystemVarName.SITE_NAME: workspace.site_name,
            SystemVarName.WORKSPACE: workspace.get_root_dir(),
            SystemVarName.JOB_CUSTOM_DIR: workspace.get_app_custom_dir(job_id),
            SystemVarName.JOB_CONFIG_DIR: workspace.get_app_config_dir(job_id),
        }
    )
    folder = variables.get("config_folder", "")
    path = os.path.join(workspace.get_app_dir(job_id), folder, "config_fed_client.json")
    config = ConfigFactory.load_config(path)
    return config is not None and execution_lifetime(config.to_dict(), variables) == ExecutionLifetime.TASK


def app_requires_task_runtime(app_data):
    # Match the supported client config basenames without extraction or imports.
    # Scan all candidates conservatively: variable resolution may depend on the
    # target site or --set values unavailable to the server at deploy time.
    with ZipFile(io.BytesIO(app_data)) as archive:
        for extension, fmt in ConfigFormat.config_ext_formats().items():
            basename = "config_fed_client" + extension
            for name in archive.namelist():
                if name.rsplit("/", 1)[-1] != basename:
                    continue
                loader = ConfigFactory.get_config_loader(fmt)
                if loader is None:
                    raise ValueError(f"unsupported client configuration format {fmt}")
                config = loader.load_config_from_str(archive.read(name).decode("utf-8")).to_dict()
                # Any variable-dependent lifetime requires the new runtime,
                # even when its submitted default is job: client --set values
                # can change it to task after this server-side preflight.
                if config.get("execution_lifetime", ExecutionLifetime.JOB) != ExecutionLifetime.JOB:
                    return True
    return False
