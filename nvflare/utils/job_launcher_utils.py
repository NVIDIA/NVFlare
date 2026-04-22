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
import logging
import os
import sys

from nvflare.apis.fl_constant import FLContextKey, SystemVarName
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


_LAUNCHER_SPEC_DEFAULT_KEY = "default"

# "default" is the only reserved top-level key in launcher_spec. Every other
# top-level key is treated as a site name. A typo such as "defaults" would be
# silently accepted as a site name and never matched during resolution.
_LAUNCHER_SPEC_RESERVED_KEYS = {_LAUNCHER_SPEC_DEFAULT_KEY}


def _validate_launcher_spec(launcher_spec: dict) -> list:
    """Return top-level keys that look like misspellings of a reserved token.

    Reserved keys (_LAUNCHER_SPEC_RESERVED_KEYS) are skipped. All other keys
    are treated as site names. A key whose sub-keys are all valid launcher modes
    but whose name closely resembles a reserved token is flagged so callers can
    warn the user before resolution silently ignores it.
    """
    suspicious = []
    for key in launcher_spec:
        if key in _LAUNCHER_SPEC_RESERVED_KEYS:
            continue
        value = launcher_spec[key]
        if not isinstance(value, dict):
            continue
        # Flag keys whose sub-keys look like launcher modes (i.e. look like a
        # site block) but whose name is a near-match for a reserved token.
        if set(value.keys()) <= _LAUNCHER_MODE_KEYS:
            for reserved in _LAUNCHER_SPEC_RESERVED_KEYS:
                if key != reserved and (key.startswith(reserved) or reserved.startswith(key)):
                    suspicious.append(key)
    return suspicious


def get_job_launcher_spec(job_meta, site_name, mode):
    """Get launcher-specific config for a site/mode.

    Resolution order:
    1. Merge launcher_spec["default"][mode] with launcher_spec[site][mode] (site wins).
    2. Fall back to get_launcher_resource_spec (nested resource_spec backward compat) when
       neither launcher_spec["default"][mode] nor launcher_spec[site][mode] is present —
       even if launcher_spec exists for other sites or modes.

    Returns a dict for the given mode, or an empty dict if not specified.
    """
    launcher_spec = job_meta.get(JobMetaKey.JOB_LAUNCHER_SPEC.value, {}) or {}
    for bad_key in _validate_launcher_spec(launcher_spec):
        logging.getLogger(__name__).warning(
            f"launcher_spec key '{bad_key}' looks like a misspelling of the reserved key "
            f"'{_LAUNCHER_SPEC_DEFAULT_KEY}' and will be treated as a site name, not a default block."
        )
    default_spec = (launcher_spec.get(_LAUNCHER_SPEC_DEFAULT_KEY) or {}).get(mode) or {}
    site_spec = (launcher_spec.get(site_name) or {}).get(mode)
    if default_spec or site_spec is not None:
        return {**default_spec, **(site_spec or {})}
    return get_launcher_resource_spec(job_meta, site_name, mode)


def add_custom_dir_to_path(app_custom_folder, new_env):
    """Util method to add app_custom_folder into the sys.path and carry into the child process."""
    sys_path = copy.copy(sys.path)
    sys_path.append(app_custom_folder)
    new_env[SystemVarName.PYTHONPATH] = os.pathsep.join(sys_path)
