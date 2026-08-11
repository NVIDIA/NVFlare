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
import importlib
import logging
import os
import re
import sys

from nvflare.apis.fl_constant import FLContextKey, SystemVarName
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobProcessEnv

# Bootstrap credentials are delivered via the job process environment (JobProcessEnv),
# never rendered into command lines. See docs/design/job_process_credential_transport_design.md.
_CREDENTIAL_ARG_ENV_NAMES = {
    JobProcessArgs.AUTH_TOKEN: JobProcessEnv.AUTH_TOKEN,
    JobProcessArgs.TOKEN_SIGNATURE: JobProcessEnv.TOKEN_SIGNATURE,
    JobProcessArgs.SSID: JobProcessEnv.SSID,
}


def get_credential_env(job_args: dict) -> dict:
    """Map bootstrap credentials in JOB_PROCESS_ARGS to their env var names.

    Empty/None credential values are skipped so the job process parser fails loudly at
    startup instead of booting with a bogus credential.
    """
    result = {}
    for arg, env in _CREDENTIAL_ARG_ENV_NAMES.items():
        e = job_args.get(arg)
        if e and e[1]:
            result[env] = str(e[1])
    return result


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
            JobProcessArgs.PARENT_URL,
            JobProcessArgs.PARENT_CONN_SEC,
            JobProcessArgs.ROOT_URL,
            JobProcessArgs.SERVICE_HOST,
            JobProcessArgs.SERVICE_PORT,
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


_LAUNCHER_MODE_KEYS = {"process", "docker", "k8s", "slurm"}

PORTABLE_RESOURCE_DEFAULT_KEY = "@default"
PORTABLE_RESOURCE_KEYS = ("num_of_gpus", "num_of_cpus", "memory")
_PORTABLE_MEMORY_PATTERN = re.compile(r"^([1-9][0-9]*)(Mi|Gi|Ti)$")
_MEMORY_UNIT_TO_MIB = {"Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
_PORTABLE_NATIVE_RESOURCE_KEYS = {
    "docker": {"num_of_cpus": {"nano_cpus"}, "memory": {"mem_limit"}},
    "k8s": {"num_of_cpus": {"cpu", "cpu_request"}, "memory": {"memory", "memory_request"}},
    "slurm": {"num_of_cpus": {"cpus_per_node"}, "memory": {"mem_per_node"}},
}


def get_site_launcher_spec(site_spec, mode):
    """Extract the launcher-mode portion of a single site's resource spec.

    New nested format: ``{mode: {...}}`` — returns the inner dict for *mode*.
    Legacy flat format: ``{num_of_gpus: ...}`` — treated as process mode for
    backward compatibility; Docker, K8s, and Slurm modes receive an empty spec.
    """
    site_spec = site_spec or {}
    if any(k in site_spec for k in _LAUNCHER_MODE_KEYS):
        return site_spec.get(mode, {})
    return site_spec if mode == "process" else {}


def get_launcher_resource_spec(job_meta, site_name, mode):
    """Extract the launcher-mode resource spec for a site from full job meta."""
    resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
    return get_site_launcher_spec(resource_spec.get(site_name), mode)


def _validate_portable_values(resource_spec: dict, label: str) -> None:
    num_gpus = resource_spec.get("num_of_gpus")
    if num_gpus is not None and (isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 0):
        raise ValueError(f"{label}.num_of_gpus must be an integer greater than or equal to 0")

    num_cpus = resource_spec.get("num_of_cpus")
    if num_cpus is not None and (isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus < 1):
        raise ValueError(f"{label}.num_of_cpus must be an integer greater than or equal to 1")

    memory = resource_spec.get("memory")
    if memory is not None and (not isinstance(memory, str) or not _PORTABLE_MEMORY_PATTERN.fullmatch(memory)):
        raise ValueError(f"{label}.memory must be a positive integer followed by Mi, Gi, or Ti")


def validate_portable_resource_spec(resource_spec: dict) -> None:
    """Validate portable resource fields without restricting site-specific custom resources."""
    if not resource_spec:
        return

    default_spec = resource_spec.get(PORTABLE_RESOURCE_DEFAULT_KEY)
    if default_spec is not None:
        unknown = set(default_spec) - set(PORTABLE_RESOURCE_KEYS)
        if unknown:
            raise ValueError(
                f"resource_spec['{PORTABLE_RESOURCE_DEFAULT_KEY}'] contains unsupported field(s): {sorted(unknown)}"
            )
        _validate_portable_values(default_spec, f"resource_spec['{PORTABLE_RESOURCE_DEFAULT_KEY}']")

    has_default = default_spec is not None
    for site_name, site_spec in resource_spec.items():
        if site_name == PORTABLE_RESOURCE_DEFAULT_KEY:
            continue
        effective_site_spec = get_site_launcher_spec(site_spec, "process") if has_default else site_spec
        if not has_default and any(key in site_spec for key in _LAUNCHER_MODE_KEYS):
            # Preserve legacy nested resource_spec behavior unless @default opts in
            # to the new portable resolution contract.
            continue
        _validate_portable_values(effective_site_spec, f"resource_spec['{site_name}']")


def resolve_site_resource_spec(job_meta: dict, site_name: str) -> dict:
    """Resolve scheduler-facing resources for a site without mutating job metadata."""
    resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
    default_spec = resource_spec.get(PORTABLE_RESOURCE_DEFAULT_KEY) or {}
    site_spec = get_site_launcher_spec(resource_spec.get(site_name), "process")
    return {**default_spec, **site_spec}


def get_portable_resource_spec(job_meta: dict, site_name: str) -> dict:
    """Return the portable fields that launchers must enforce for a site."""
    resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
    site_spec = resource_spec.get(site_name) or {}
    if PORTABLE_RESOURCE_DEFAULT_KEY in resource_spec:
        resolved = resolve_site_resource_spec(job_meta, site_name)
    elif any(key in site_spec for key in _LAUNCHER_MODE_KEYS):
        # Legacy nested specs are launcher-specific, not portable.
        resolved = {}
    else:
        resolved = site_spec
    portable = {key: resolved[key] for key in PORTABLE_RESOURCE_KEYS if key in resolved}
    _validate_portable_values(portable, f"resource_spec for site '{site_name}'")
    return portable


def portable_memory_to_mib(memory: str) -> int:
    """Convert a validated portable memory value to an exact MiB count."""
    match = _PORTABLE_MEMORY_PATTERN.fullmatch(memory) if isinstance(memory, str) else None
    if not match:
        raise ValueError("memory must be a positive integer followed by Mi, Gi, or Ti")
    value, unit = match.groups()
    return int(value) * _MEMORY_UNIT_TO_MIB[unit]


def portable_memory_to_bytes(memory: str) -> int:
    return portable_memory_to_mib(memory) * 1024 * 1024


def validate_portable_resource_conflicts(job_meta: dict) -> None:
    """Reject simultaneous portable and equivalent launcher-native CPU or memory fields."""
    resource_spec = job_meta.get(JobMetaKey.RESOURCE_SPEC.value, {}) or {}
    launcher_spec = job_meta.get(JobMetaKey.JOB_LAUNCHER_SPEC.value, {}) or {}
    site_names = (set(resource_spec) - {PORTABLE_RESOURCE_DEFAULT_KEY}) | (set(launcher_spec) - {"default"})
    for site_name in site_names:
        portable = get_portable_resource_spec(job_meta, site_name)
        for mode, portable_to_native in _PORTABLE_NATIVE_RESOURCE_KEYS.items():
            native = get_job_launcher_spec(job_meta, site_name, mode)
            for portable_key, native_keys in portable_to_native.items():
                conflicts = sorted(native_keys & set(native)) if portable_key in portable else []
                if conflicts:
                    raise ValueError(
                        f"portable resource '{portable_key}' conflicts with launcher_spec {mode} field(s) "
                        f"{conflicts} for site '{site_name}'"
                    )

    default_portable = resource_spec.get(PORTABLE_RESOURCE_DEFAULT_KEY) or {}
    default_launcher = launcher_spec.get("default") or {}
    for mode, portable_to_native in _PORTABLE_NATIVE_RESOURCE_KEYS.items():
        native = default_launcher.get(mode) or {}
        for portable_key, native_keys in portable_to_native.items():
            conflicts = sorted(native_keys & set(native)) if portable_key in default_portable else []
            if conflicts:
                raise ValueError(
                    f"portable resource '{portable_key}' conflicts with launcher_spec {mode} field(s) "
                    f"{conflicts} in the default blocks"
                )


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


def refresh_custom_dir_import_path(app_custom_folder):
    """Refresh import state after a job custom dir is created post-interpreter startup."""
    if not app_custom_folder:
        return
    if not os.path.isdir(app_custom_folder):
        logging.getLogger(__name__).debug(
            "refresh_custom_dir_import_path: custom dir not found, skipping: %s", app_custom_folder
        )
        return
    if app_custom_folder not in sys.path:
        sys.path.append(app_custom_folder)
    importlib.invalidate_caches()
