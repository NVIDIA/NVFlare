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
import shlex
from typing import Optional

from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME, JobMetaKey
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType  # noqa: F401 - public re-export
from nvflare.fuel.utils.secret_utils import has_secret_refs, split_command_preserving_secret_refs
from nvflare.utils.argv_utils import CommandArg, normalize_argv

from .api import FedJob

_ADDITIONAL_NODE_COMMAND = "additional_node_command"


def _to_external_process_argv(value: CommandArg, arg_name: str) -> list[str]:
    """Return shell-free argv while preserving pre-tokenized values exactly."""
    normalized = normalize_argv(value, arg_name)
    assert normalized is not None
    if isinstance(normalized, str):
        return split_command_preserving_secret_refs(normalized, posix=True)
    return normalized


def _fill_additional_node_command(job: FedJob, target: str, command: list[str], launch_once: bool) -> None:
    """Copy a managed external-process command into explicit multi-node launcher blocks."""
    meta_props = job.job.meta_props
    launcher_spec = meta_props.get(JobMetaKey.JOB_LAUNCHER_SPEC.value) if isinstance(meta_props, dict) else None
    if not isinstance(launcher_spec, dict):
        return

    command_text = None
    for site_name, site_spec in launcher_spec.items():
        if (
            site_name in ("default", SERVER_SITE_NAME)
            or (target != ALL_SITES and site_name != target)
            or not isinstance(site_spec, dict)
        ):
            continue
        for block in site_spec.values():
            nodes = block.get("nodes") if isinstance(block, dict) else None
            if not isinstance(nodes, int) or nodes <= 1 or _ADDITIONAL_NODE_COMMAND in block:
                continue
            if not launch_once:
                raise ValueError("generated additional_node_command requires launch_once=True")
            if command_text is None:
                if any(has_secret_refs(arg) for arg in command):
                    raise ValueError(
                        "additional_node_command does not support secret references; set a secret-free command explicitly"
                    )
                command_text = shlex.join(command)
            block[_ADDITIONAL_NODE_COMMAND] = command_text


class ScriptRunner:
    """Adds a Client API training script to a FedJob.

    Transport is selected by the site's Cell driver configuration. The runner only
    selects whether the trainer runs in the Client Job process or in a process owned
    and launched by NVFlare.
    """

    def __init__(
        self,
        script: str,
        script_args: Optional[CommandArg] = "",
        launch_external_process: bool = False,
        command: CommandArg = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        launch_once: bool = True,
        launch_timeout: Optional[float] = 300.0,
        shutdown_timeout: float = 0.0,
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        execution_mode: Optional[str] = None,
    ):
        """Initializes the runner.

        Args:
            script: Training script path.
            script_args: Arguments appended to the script. Pre-tokenized argv preserves
                exact argument boundaries in both execution modes.
            launch_external_process: Select ``external_process`` when ``execution_mode``
                is omitted; otherwise select ``in_process``.
            command: Command prepended to the script in ``external_process`` mode.
            framework: Trainer-native parameter representation.
            server_expected_format: Parameter representation expected by the server.
            params_transfer_type: Whether the trainer returns FULL parameters or a DIFF.
            launch_once: Launch once per job or once per task in ``external_process`` mode.
            launch_timeout: Maximum time for the external trainer to initialize and connect.
            shutdown_timeout: External-process orderly-exit wait. This also feeds several
                accepted-result cleanup bounds in ``ClientAPIExecutor``. The default zero skips
                direct orderly-exit and finalize-gate waits, but maps to 30 seconds for accepted-
                source disconnect and post-settlement group-exit waits and feeds the fixed settled-
                reaper budget; see ``ClientAPIExecutor.shutdown_timeout`` for all roles.
            memory_gc_rounds: Force memory cleanup every N rounds; zero disables it.
            cuda_empty_cache: Empty the CUDA cache during configured memory cleanup.
            execution_mode: Optional explicit ``in_process`` or ``external_process`` mode.
                Use ``ClientAPIExecutor`` directly for an independently managed trainer
                in ``attach`` mode.
        """
        if execution_mode is None:
            execution_mode = ExecutionMode.EXTERNAL_PROCESS if launch_external_process else ExecutionMode.IN_PROCESS
        available_modes = (ExecutionMode.IN_PROCESS, ExecutionMode.EXTERNAL_PROCESS)
        if execution_mode not in available_modes:
            raise ValueError(
                f"invalid execution_mode {execution_mode!r} for ScriptRunner: "
                f"must be one of {list(available_modes)}; use ClientAPIExecutor directly for attach mode"
            )
        if launch_external_process and execution_mode != ExecutionMode.EXTERNAL_PROCESS:
            raise ValueError(
                "launch_external_process=True requires execution_mode='external_process', "
                f"but got execution_mode={execution_mode!r}"
            )

        format_by_framework = {
            FrameworkType.PYTORCH: ExchangeFormat.PYTORCH,
            FrameworkType.TENSORFLOW: ExchangeFormat.KERAS_LAYER_WEIGHTS,
            FrameworkType.NUMPY: ExchangeFormat.NUMPY,
            FrameworkType.RAW: ExchangeFormat.RAW,
        }
        params_exchange_format = format_by_framework.get(framework)
        if params_exchange_format is None:
            raise ValueError(f"Framework {framework} unsupported")

        self._script = script
        normalized_script_args = normalize_argv(script_args, "script_args", allow_none=True)
        self._script_args = "" if normalized_script_args is None else normalized_script_args
        normalized_command = normalize_argv(command, "command")
        assert normalized_command is not None
        self._command = normalized_command
        self._launch_external_process = execution_mode == ExecutionMode.EXTERNAL_PROCESS
        self._server_expected_format = server_expected_format
        self._framework = framework
        self._params_transfer_type = params_transfer_type
        self._launch_once = launch_once
        self._launch_timeout = launch_timeout
        self._shutdown_timeout = shutdown_timeout
        self._memory_gc_rounds = memory_gc_rounds
        self._cuda_empty_cache = cuda_empty_cache
        self._execution_mode = execution_mode
        self._params_exchange_format = params_exchange_format

    def _external_process_argv(self, packaged_script_path: Optional[str]) -> list[str]:
        command = _to_external_process_argv(self._command, "command")
        # Preserve the established external-process command for an absolute script
        # that is expected to exist only on the target production client.
        script = packaged_script_path
        if script is None:
            script = os.path.basename(self._script) if os.path.isabs(self._script) else self._script
        command.append(f"custom/{script}")
        command.extend(_to_external_process_argv(self._script_args, "script_args"))
        return command

    def _packaged_script_path(self, job: FedJob) -> Optional[str]:
        """Choose a stable client-relative path only when this process can bundle the script."""
        # Match FedApp._add_resource(), which intentionally gives directories
        # precedence when a filesystem abstraction reports both classifications.
        if os.path.isdir(self._script):
            return None
        if not os.path.isfile(self._script):
            return None
        if os.path.isabs(self._script):
            # Freeze the exporter's established sys.path-relative destination now.
            # The authoring environment may change before export, and separate source
            # directories must not collapse to the same basename.
            return job.job._get_relative_script(self._script)
        return self._script

    def add_to_fed_job(self, job: FedJob, ctx, **kwargs):
        """Adds the configured ClientAPIExecutor and script resource to the job."""
        job.check_kwargs(args_to_check=kwargs, args_expected={"tasks": False})
        tasks = kwargs.get("tasks", ["*"])
        packaged_script_path = self._packaged_script_path(job)

        common_args = {
            "execution_mode": self._execution_mode,
            "params_exchange_format": self._params_exchange_format,
            "server_expected_format": self._server_expected_format,
            "params_transfer_type": self._params_transfer_type,
            "memory_gc_rounds": self._memory_gc_rounds,
            "cuda_empty_cache": self._cuda_empty_cache,
        }
        if self._execution_mode == ExecutionMode.EXTERNAL_PROCESS:
            command = self._external_process_argv(packaged_script_path)
            _fill_additional_node_command(job, ctx.target, command, self._launch_once)
            executor = ClientAPIExecutor(
                command=command,
                launch_once=self._launch_once,
                launch_timeout=self._launch_timeout,
                shutdown_timeout=self._shutdown_timeout,
                **common_args,
            )
        else:
            # Locally available absolute paths identify authoring-machine source files,
            # so execute their bundled relative path. An unavailable absolute path is a
            # supported reference to a script pre-installed on a production client.
            task_script_path = packaged_script_path or self._script
            executor = ClientAPIExecutor(
                task_script_path=task_script_path,
                task_script_args=self._script_args,
                **common_args,
            )

        job.add_executor(executor, tasks=tasks, ctx=ctx)
        job.add_resources(resources=[self._script], ctx=ctx)
        if packaged_script_path is not None:
            # ScriptRunner and the exporter are part of the same job-config layer. Store
            # the selected destination privately so packaging cannot derive a different
            # path later from a changed working directory or sys.path.
            app_config = job._get_app(ctx).app_config
            app_config._set_ext_script_destination(self._script, packaged_script_path)
        return {}
