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
"""Render Slurm batch scripts and submission arguments."""

from __future__ import annotations

import os
import re
import shlex

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.app_opt.job_launcher.slurm.config import (
    BATCH_FILE,
    NODE_FILE,
    SECRET_FILE,
    SLURM_CHILD_PROCESS_ENV,
    SLURM_SBATCH_DIRECTIVES,
    LaunchPlan,
    SlurmConfig,
)
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR, CELL_API_TYPE, bootstrap_file_name

_ENV_NNODES = "NVFL_NNODES"
_ENV_NODE_RANK = "NVFL_NODE_RANK"
_ENV_MASTER_ADDR = "NVFL_MASTER_ADDR"
_ENV_MASTER_PORT = "NVFL_MASTER_PORT"
_ENV_RUN_ID = "NVFL_RUN_ID"
_MULTINODE_BATCH_ENV = (_ENV_NNODES, _ENV_MASTER_ADDR, _ENV_MASTER_PORT, _ENV_RUN_ID)

_TEMPLATE_TOKEN_PATTERN = re.compile(r"@@NVFLARE_[A-Z0-9_]+@@")

_BATCH_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

_nvfl_secret=@@NVFLARE_SECRET_PATH@@
_nvfl_cleanup() {
  _nvfl_status=$?
  rm -f -- "${_nvfl_secret}" || true
  return "${_nvfl_status}"
}
trap _nvfl_cleanup EXIT

[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || {
  echo "requeued NVFlare job refused" >&2
  exit 101
}

@@NVFLARE_COMMON_ENVIRONMENT@@
@@NVFLARE_BACKEND_ENVIRONMENT@@
_nvfl_command=(@@NVFLARE_COMMAND@@)
exec "${_nvfl_command[@]}"
"""

_NODE_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

export NVFL_NODE_RANK="$((10#${SLURM_NODEID}))"

@@NVFLARE_BACKEND_SETUP@@
if [[ "${NVFL_NODE_RANK}" == "0" ]]; then
  _nvfl_command=(@@NVFLARE_RANK0_COMMAND@@)
else
@@NVFLARE_NONZERO_SETUP@@
  _nvfl_command=(@@NVFLARE_NONZERO_COMMAND@@)
fi

exec "${_nvfl_command[@]}"
"""


def _render_shell_template(source: str, **values) -> str:
    replacements = {f"@@NVFLARE_{key.upper()}@@": str(value) for key, value in values.items()}
    return _TEMPLATE_TOKEN_PATTERN.sub(lambda match: replacements[match.group(0)], source)


def _build_worker_words(plan: LaunchPlan) -> list[str]:
    return [shlex.quote(value) for value in (plan.python_path, "-u", "-m", plan.exe_module, *plan.module_args)]


def _tool_assignment(variable: str, configured: str | None, default: str) -> str:
    return f"{variable}={shlex.quote(configured or default)}"


def _common_environment(plan: LaunchPlan, config: SlurmConfig) -> list[str]:
    lines = ['[[ -r "${_nvfl_secret}" ]] || { echo "NVFlare Slurm secret file is unavailable" >&2; exit 100; }']
    if plan.setup:
        lines.extend(["export SLURM_EXPORT_ENV=ALL", plan.setup])
    lines.extend(
        [
            # Restore this after trusted setup in case setup changed it.
            "export SLURM_EXPORT_ENV=ALL",
            "set +x",
            'source "${_nvfl_secret}"',
            'rm -f -- "${_nvfl_secret}"',
            # Bash reports status 0 to an EXIT trap when exec itself fails.
            "trap - EXIT",
            f"export PYTHONPATH={shlex.quote(plan.python_env)}",
            f"export {SLURM_CHILD_PROCESS_ENV}=1",
        ]
    )
    for name, value in sorted(plan.study_env.items()):
        lines.append(f"export {name}={shlex.quote(value)}")
    for name in plan.forward_env:
        lines.append(f"if [[ ${{{name}+x}} ]]; then export {name}; fi")
    if plan.sandbox == "pyxis" or plan.additional_node_command:
        lines.append(_tool_assignment("NVFL_SRUN", config.executables.get("srun"), "srun"))
    return lines


def _apptainer_environment(plan: LaunchPlan, config: SlurmConfig) -> list[str]:
    environment = [
        _tool_assignment("NVFL_APPTAINER", config.executables.get("apptainer"), "apptainer"),
        "for _nvfl_name in ${!APPTAINER_@} ${!APPTAINERENV_@} "
        '${!SINGULARITY_@} ${!SINGULARITYENV_@}; do unset "${_nvfl_name}"; done',
        'export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"',
        f"export APPTAINERENV_{SLURM_CHILD_PROCESS_ENV}=1",
    ]
    for name in sorted(set(plan.study_env) | set(plan.study_secret_env)):
        environment.append(f'export APPTAINERENV_{name}="${{{name}}}"')
    for name in plan.forward_env:
        environment.append(f'if [[ ${{{name}+x}} ]]; then export APPTAINERENV_{name}="${{{name}}}"; fi')
    return environment


def _apptainer_exec_words(plan: LaunchPlan, pwd: str) -> list[str]:
    command = ['"${NVFL_APPTAINER}"']
    command.extend(
        shlex.quote(word)
        for word in (
            "exec",
            "--userns",
            "--containall",
            "--no-eval",
            "--no-privs",
            "--no-mount",
            "bind-paths,hostfs",
        )
    )
    for mount in plan.mounts:
        command.extend((shlex.quote("--bind"), shlex.quote(mount.render())))
    command.extend((shlex.quote("--pwd"), shlex.quote(pwd)))
    if plan.resources.gpus_per_node:
        command.append(shlex.quote("--nv"))
    return command + [shlex.quote(plan.image)]


def _apptainer_parts(plan: LaunchPlan, config: SlurmConfig, worker_words: list[str]) -> tuple[list[str], list[str]]:
    return _apptainer_environment(plan, config), _apptainer_exec_words(plan, plan.run_dir) + worker_words


def _multinode_srun_words(plan: LaunchPlan) -> list[str]:
    return [
        '"${NVFL_SRUN}"',
        shlex.quote(f"--nodes={plan.resources.nodes}"),
        shlex.quote(f"--ntasks={plan.resources.nodes}"),
        shlex.quote("--ntasks-per-node=1"),
        # Any failing task terminates the whole step so the allocation cannot idle
        # until wall time; --wait=0 waits indefinitely after clean task exits so a
        # finished worker never kills a still-running rank 0.
        shlex.quote("--kill-on-bad-exit=1"),
        shlex.quote("--wait=0"),
        shlex.quote("--label"),
    ]


def _multinode_parts(plan: LaunchPlan, job_dir: str, config: SlurmConfig) -> tuple[list[str], list[str]]:
    port_start, port_end = config.multi_node_port_range
    port_count = port_end - port_start + 1
    environment = [
        f'export {_ENV_NNODES}="${{SLURM_JOB_NUM_NODES:?}}"',
        # The batch script always executes on the first node of the allocation.
        f'export {_ENV_MASTER_ADDR}="${{SLURMD_NODENAME:?}}"',
        f'export {_ENV_MASTER_PORT}="$(({port_start} + 10#${{SLURM_JOB_ID}} % {port_count}))"',
        f'export {_ENV_RUN_ID}="${{SLURM_JOB_ID:?}}"',
    ]
    node_script = shlex.quote(os.path.join(job_dir, NODE_FILE))
    container_words = []
    if plan.sandbox == "apptainer":
        environment.extend(_apptainer_environment(plan, config))
        environment.extend(f'export APPTAINERENV_{name}="${{{name}}}"' for name in _MULTINODE_BATCH_ENV)
    elif plan.sandbox == "pyxis":
        environment.extend(_pyxis_environment(plan, extra_names=_MULTINODE_BATCH_ENV))
        container_words = _pyxis_container_words(plan)
    return environment, _multinode_srun_words(plan) + container_words + [node_script]


def _render_node_script(plan: LaunchPlan, config: SlurmConfig) -> str:
    """Render the per-node dispatch script for a launcher-owned node group.

    For bare mode it runs on the host; for Pyxis it runs inside the per-task
    container that srun already created; for Apptainer it runs on the host and
    starts the per-node container itself.
    """
    worker_words = _build_worker_words(plan)
    node_words = [shlex.quote(word) for word in plan.additional_node_command]
    credential_names = (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID)
    nonzero_setup = [
        f"  unset {' '.join(credential_names)}",
        f"  export {CLIENT_API_TYPE_KEY}={CELL_API_TYPE}",
        f"  export {BOOTSTRAP_FILE_ENV_VAR}={shlex.quote(bootstrap_file_name(1))}",
    ]
    if plan.sandbox == "apptainer":
        backend_setup = "\n".join(
            (
                _tool_assignment("NVFL_APPTAINER", config.executables.get("apptainer"), "apptainer"),
                f'export APPTAINERENV_{_ENV_NODE_RANK}="${{{_ENV_NODE_RANK}}}"',
            )
        )
        rank0_command = _apptainer_exec_words(plan, plan.run_dir) + worker_words
        nonzero_setup.append(f"  unset {' '.join(f'APPTAINERENV_{name}' for name in credential_names)}")
        nonzero_setup.append(f'  export APPTAINERENV_{CLIENT_API_TYPE_KEY}="${{{CLIENT_API_TYPE_KEY}}}"')
        nonzero_setup.append(f'  export APPTAINERENV_{BOOTSTRAP_FILE_ENV_VAR}="${{{BOOTSTRAP_FILE_ENV_VAR}}}"')
        nonzero_command = _apptainer_exec_words(plan, plan.node_app_dir) + node_words
    else:
        backend_setup = ""
        rank0_command = worker_words
        nonzero_setup.append(f"  cd {shlex.quote(plan.node_app_dir)}")
        nonzero_command = node_words
    return _render_shell_template(
        _NODE_SCRIPT_TEMPLATE,
        backend_setup=backend_setup,
        rank0_command=" ".join(rank0_command),
        nonzero_setup="\n".join(nonzero_setup),
        nonzero_command=" ".join(nonzero_command),
    )


def _pyxis_environment(plan: LaunchPlan, extra_names: tuple = ()) -> list[str]:
    fixed_names = sorted(
        {"PYTHONPATH", SLURM_CHILD_PROCESS_ENV, *extra_names} | set(plan.study_env) | set(plan.study_secret_env)
    )
    environment = [f"_nvfl_container_env={shlex.quote(','.join(fixed_names))}"]
    for name in plan.forward_env:
        environment.append(f'if [[ ${{{name}+x}} ]]; then _nvfl_container_env="${{_nvfl_container_env}},{name}"; fi')
    return environment


def _pyxis_container_words(plan: LaunchPlan) -> list[str]:
    mounts = ",".join(
        f"{mount.source}:{mount.destination}" if mount.mode == "rw" else mount.render() for mount in plan.mounts
    )
    return [
        '"--export=${_nvfl_container_env}"',
        shlex.quote(f"--container-image={plan.image}"),
        shlex.quote("--container-readonly"),
        shlex.quote("--no-container-mount-home"),
        shlex.quote("--no-container-entrypoint"),
        shlex.quote(f"--container-workdir={plan.run_dir}"),
        shlex.quote(f"--container-mounts={mounts}"),
        '"--container-env=${_nvfl_container_env}"',
    ]


def _pyxis_parts(plan: LaunchPlan, worker_words: list[str]) -> tuple[list[str], list[str]]:
    environment = _pyxis_environment(plan)
    command = ['"${NVFL_SRUN}"', shlex.quote("--ntasks=1")] + _pyxis_container_words(plan) + worker_words
    return environment, command


def _render_batch_script(
    plan: LaunchPlan,
    job_dir: str,
    config: SlurmConfig,
) -> tuple[str, dict]:
    secret_values = plan.study_secret_env
    secret_path = os.path.join(job_dir, SECRET_FILE)
    if plan.additional_node_command:
        environment, command_words = _multinode_parts(plan, job_dir, config)
    elif plan.sandbox == "apptainer":
        environment, command_words = _apptainer_parts(plan, config, _build_worker_words(plan))
    elif plan.sandbox == "pyxis":
        environment, command_words = _pyxis_parts(plan, _build_worker_words(plan))
    else:
        environment = []
        command_words = _build_worker_words(plan)
    script = _render_shell_template(
        _BATCH_SCRIPT_TEMPLATE,
        secret_path=shlex.quote(secret_path),
        common_environment="\n".join(_common_environment(plan, config)),
        backend_environment="\n".join(environment),
        command=" ".join(command_words),
    )
    return script, secret_values


def _render_secret_file(values: dict) -> str:
    lines = ["# generated transient NVFlare secrets"]
    for name, value in sorted(values.items()):
        lines.append(f"export {name}={shlex.quote(value)}")
    lines.append("")
    return "\n".join(lines)


def _submission_argv(plan: LaunchPlan, job_dir: str, job_name: str, marker: str, config: SlurmConfig) -> list[str]:
    argv = [
        config.executables["sbatch"],
        "--parsable",
        "--no-requeue",
        "--export=NIL",
        f"--chdir={plan.run_dir}",
        f"--nodes={plan.resources.nodes}",
        f"--ntasks={plan.resources.nodes}",
        "--ntasks-per-node=1",
    ]
    if plan.resources.gpus_per_node:
        argv.append(f"--gres=gpu:{plan.resources.gpus_per_node}")
    if plan.resources.cpus_per_node:
        argv.append(f"--cpus-per-task={plan.resources.cpus_per_node}")
    if plan.resources.mem_per_node:
        argv.append(f"--mem={plan.resources.mem_per_node}M")
    directives = dict(plan.directives)
    if plan.resources.time_limit:
        directives["time"] = plan.resources.time_limit
    for key in SLURM_SBATCH_DIRECTIVES:
        value = directives.get(key)
        if value is not None:
            argv.append(f"--{key}={value}")
    argv.extend(
        [
            f"--job-name={job_name}",
            f"--comment={marker}",
            f"--output={os.path.join(plan.run_dir, 'slurm-%j.out')}",
            os.path.join(job_dir, BATCH_FILE),
        ]
    )
    return argv
