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
import shlex

from nvflare.apis.job_launcher_spec import CREDENTIAL_ENV_NAMES
from nvflare.app_common.multinode import (
    CONTRACT_ENV_NAMES,
    ENV_MASTER_ADDR,
    ENV_MASTER_PORT,
    ENV_NNODES,
    ENV_NODE_RANK,
    ENV_RUN_ID,
)
from nvflare.app_opt.job_launcher.slurm.config import (
    BATCH_FILE,
    NODE_FILE,
    SECRET_FILE,
    SLURM_CHILD_PROCESS_ENV,
    SLURM_SBATCH_DIRECTIVES,
    LaunchPlan,
    SlurmConfig,
)


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
    if plan.sandbox == "pyxis":
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


# ENV_NODE_RANK is exported per task by node.sh; the rest are batch-level.
_MULTINODE_BATCH_ENV = tuple(name for name in CONTRACT_ENV_NAMES if name != ENV_NODE_RANK)


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
        _tool_assignment("NVFL_SRUN", config.executables.get("srun"), "srun"),
        '[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid SLURM_JOB_ID" >&2; exit 102; }',
        f'export {ENV_NNODES}="${{SLURM_JOB_NUM_NODES:?}}"',
        # The batch script always executes on the first node of the allocation.
        f'export {ENV_MASTER_ADDR}="${{SLURMD_NODENAME:?}}"',
        f'export {ENV_MASTER_PORT}="$(({port_start} + 10#${{SLURM_JOB_ID}} % {port_count}))"',
        f'export {ENV_RUN_ID}="${{SLURM_JOB_ID}}"',
    ]
    node_script = shlex.quote(os.path.join(job_dir, NODE_FILE))
    if plan.sandbox == "apptainer":
        environment.extend(_apptainer_environment(plan, config))
        environment.extend(f'export APPTAINERENV_{name}="${{{name}}}"' for name in _MULTINODE_BATCH_ENV)
        command = _multinode_srun_words(plan) + [node_script]
    elif plan.sandbox == "pyxis":
        environment.extend(_pyxis_environment(plan, extra_names=_MULTINODE_BATCH_ENV))
        command = _multinode_srun_words(plan) + _pyxis_container_words(plan) + [node_script]
    else:
        command = _multinode_srun_words(plan) + [node_script]
    return environment, command


def _render_node_script(plan: LaunchPlan, config: SlurmConfig) -> str:
    """Render the per-node dispatch script for a launcher-owned node group.

    For bare mode it runs on the host; for Pyxis it runs inside the per-task
    container that srun already created; for Apptainer it runs on the host and
    starts the per-node container itself.
    """
    worker_words = _build_worker_words(plan)
    node_words = [shlex.quote(word) for word in plan.node_command]
    nodes = plan.resources.nodes
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'[[ "${{SLURM_JOB_NUM_NODES:-}}" == "{nodes}" && "${{SLURM_NODEID:-}}" =~ ^[0-9]+$ ]] '
        '|| { echo "node group topology mismatch" >&2; exit 103; }',
        f'(( 10#${{SLURM_NODEID}} < {nodes} )) || {{ echo "node rank outside node group" >&2; exit 103; }}',
        f'export {ENV_NODE_RANK}="${{SLURM_NODEID}}"',
    ]
    if plan.sandbox == "apptainer":
        # Bootstrap credentials are for the rank-0 CJ only (JobProcessEnv contract).
        unset_names = CREDENTIAL_ENV_NAMES + tuple(
            f"{prefix}{name}" for prefix in ("APPTAINERENV_", "SINGULARITYENV_") for name in CREDENTIAL_ENV_NAMES
        )
        lines.extend(
            [
                _tool_assignment("NVFL_APPTAINER", config.executables.get("apptainer"), "apptainer"),
                f'export APPTAINERENV_{ENV_NODE_RANK}="${{{ENV_NODE_RANK}}}"',
                f'if [[ "${{{ENV_NODE_RANK}}}" == "0" ]]; then',
                f"  _nvfl_command=({' '.join(_apptainer_exec_words(plan, plan.run_dir) + worker_words)})",
                "else",
                f"  unset {' '.join(unset_names)}",
                f"  _nvfl_command=({' '.join(_apptainer_exec_words(plan, plan.node_app_dir) + node_words)})",
                "fi",
            ]
        )
    else:
        lines.extend(
            [
                f'if [[ "${{{ENV_NODE_RANK}}}" == "0" ]]; then',
                f"  _nvfl_command=({' '.join(worker_words)})",
                "else",
                f"  unset {' '.join(CREDENTIAL_ENV_NAMES)}",
                f"  cd {shlex.quote(plan.node_app_dir)}",
                f"  _nvfl_command=({' '.join(node_words)})",
                "fi",
            ]
        )
    lines.extend(['exec "${_nvfl_command[@]}"', ""])
    return "\n".join(lines)


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
    worker_words = _build_worker_words(plan)
    secret_values = plan.study_secret_env
    secret_path = os.path.join(job_dir, SECRET_FILE)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"_nvfl_secret={shlex.quote(secret_path)}",
        '_nvfl_cleanup() { _nvfl_status=$?; rm -f -- "${_nvfl_secret}" || true; return "${_nvfl_status}"; }',
        "trap _nvfl_cleanup EXIT",
        "[[ \"${SLURM_RESTART_COUNT:-0}\" == 0 ]] || { echo 'requeued NVFlare job refused' >&2; exit 101; }",
    ]
    lines.extend(_common_environment(plan, config))
    if plan.node_command:
        environment, command_words = _multinode_parts(plan, job_dir, config)
    elif plan.sandbox == "apptainer":
        environment, command_words = _apptainer_parts(plan, config, worker_words)
    elif plan.sandbox == "pyxis":
        environment, command_words = _pyxis_parts(plan, worker_words)
    else:
        environment = []
        command_words = worker_words
    lines.extend(environment)

    lines.extend(
        [
            f"_nvfl_command=({' '.join(command_words)})",
            'exec "${_nvfl_command[@]}"',
            "",
        ]
    )
    return "\n".join(lines), secret_values


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
