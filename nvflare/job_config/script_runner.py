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

from typing import Optional, Type, Union

from nvflare.app_common.abstract.launcher import Launcher
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Pipe

from .api import FedJob, validate_object_for_job


class FrameworkType:
    RAW = "raw"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class BaseScriptRunner:
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        params_transfer_type: str = TransferType.FULL,
        executor: Union[ClientAPILauncherExecutor, InProcessClientAPIExecutor, None] = None,
        task_pipe: Optional[Pipe] = None,
        launcher: Optional[Launcher] = None,
        metric_relay: Optional[MetricRelay] = None,
        metric_pipe: Optional[Pipe] = None,
    ):
        """BaseScriptRunner is used with FedJob API to run or launch a script.

        If executor is not provided,
            `launch_external_process=False` uses InProcessClientAPIExecutor.
            `launch_external_process=True` uses ClientAPILauncherExecutor.
        else the provided executor will be used.

        If some components are passed in, it is user's responsibility to make sure they are consistent with each other.
        For example, if user provide a task_pipe, the default task_pipe_id is "task_pipe",
        please make sure the ClientAPILauncherExecutor is created using the matching pipe_id.

        Args:
            script (str): Script to run. For in-process must be a python script path. For ex-process can be any script support by `command`.
            script_args (str): Optional arguments for script (appended to script).
            launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
            command (str): If launch_external_process=True, command to run script (preprended to script). Defaults to "python3".
            framework (str): Framework type to connfigure converter and params exchange formats. Defaults to FrameworkType.PYTORCH.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent. Defaults to TransferType.FULL.
            executor (Union[ClientAPILauncherExecutor, InProcessClientAPIExecutor, None], optional):
                The executor to use in client process. Can be an instance of
                `ClientAPILauncherExecutor`, `InProcessClientAPIExecutor`, or `None`. Defaults to `None`.
                If specified, the script and script_args and command will be ignored.

            task_pipe (Optional[Pipe], optional):
                An optional Pipe instance for passing task between ClientAPILauncherExecutor
                and client api, this is only used if `launch_external_process` is True.

            launcher (Optional[Launcher], optional):
                The launcher to use with ClientAPILauncherExecutor, only used if `launch_external_process` is True.
                Defaults to `None`.

            metric_relay (Optional[MetricRelay], optional):
                An optional MetricRelay instance that can be used to relay metrics to the server.
                Defaults to `None`.

            metric_pipe (Optional[Pipe], optional):
                An optional Pipe instance for passing metric data between components. This allows
                for real-time metric handling during execution. Defaults to `None`.
        """
        self._script = script
        self._script_args = script_args
        self._command = command
        self._launch_external_process = launch_external_process
        self._framework = framework
        self._params_transfer_type = params_transfer_type

        self._params_exchange_format = None

        if self._framework == FrameworkType.PYTORCH:
            _, torch_ok = optional_import(module="torch")
            if torch_ok:
                self._params_exchange_format = ExchangeFormat.PYTORCH
            else:
                raise ValueError("Using FrameworkType.PYTORCH, but unable to import torch")
        elif self._framework == FrameworkType.TENSORFLOW:
            _, tf_ok = optional_import(module="tensorflow")
            if tf_ok:
                self._params_exchange_format = ExchangeFormat.NUMPY
            else:
                raise ValueError("Using FrameworkType.TENSORFLOW, but unable to import tensorflow")
        elif self._framework == FrameworkType.NUMPY:
            self._params_exchange_format = ExchangeFormat.NUMPY
        elif self._framework == FrameworkType.RAW:
            self._params_exchange_format = ExchangeFormat.RAW
        else:
            raise ValueError(f"Framework {self._framework} unsupported")

        if launch_external_process:
            if metric_pipe is not None:
                validate_object_for_job("metric_pipe", metric_pipe, Pipe)
            if metric_relay is not None:
                validate_object_for_job("metric_relay", metric_relay, MetricRelay)
            if task_pipe is not None:
                validate_object_for_job("task_pipe", task_pipe, Pipe)
            if executor is not None:
                validate_object_for_job("executor", executor, ClientAPILauncherExecutor)
            if launcher is not None:
                validate_object_for_job("launcher", launcher, Launcher)
        elif executor is not None:
            validate_object_for_job("executor", executor, InProcessClientAPIExecutor)

        self._metric_pipe = metric_pipe
        self._metric_relay = metric_relay
        self._task_pipe = task_pipe
        self._executor = executor
        self._launcher = launcher

    def add_to_fed_job(self, job: FedJob, ctx, **kwargs):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        job.check_kwargs(args_to_check=kwargs, args_expected={"tasks": False})
        tasks = kwargs.get("tasks", ["*"])
        comp_ids = {}

        if self._launch_external_process:
            task_pipe = (
                self._task_pipe
                if self._task_pipe
                else CellPipe(
                    mode="PASSIVE",
                    site_name="{SITE_NAME}",
                    token="{JOB_ID}",
                    root_url="{ROOT_URL}",
                    secure_mode="{SECURE_MODE}",
                    workspace_dir="{WORKSPACE}",
                )
            )
            task_pipe_id = job.add_component("pipe", task_pipe, ctx)
            comp_ids["pipe_id"] = task_pipe_id

            launcher = (
                self._launcher
                if self._launcher
                else SubprocessLauncher(
                    script=self._command + " custom/" + self._script + " " + self._script_args,
                )
            )
            launcher_id = job.add_component("launcher", launcher, ctx)
            comp_ids["launcher_id"] = launcher_id

            executor = (
                self._executor
                if self._executor
                else self._get_ex_process_executor_cls(self._framework)(
                    pipe_id=task_pipe_id,
                    launcher_id=launcher_id,
                    params_exchange_format=self._params_exchange_format,
                    params_transfer_type=self._params_transfer_type,
                    heartbeat_timeout=0,
                )
            )
            job.add_executor(executor, tasks=tasks, ctx=ctx)

            metric_pipe = (
                self._metric_pipe
                if self._metric_pipe
                else CellPipe(
                    mode="PASSIVE",
                    site_name="{SITE_NAME}",
                    token="{JOB_ID}",
                    root_url="{ROOT_URL}",
                    secure_mode="{SECURE_MODE}",
                    workspace_dir="{WORKSPACE}",
                )
            )
            metric_pipe_id = job.add_component("metrics_pipe", metric_pipe, ctx)
            comp_ids["metric_pipe_id"] = metric_pipe_id

            component = (
                self._metric_relay
                if self._metric_relay
                else MetricRelay(
                    pipe_id=metric_pipe_id,
                    event_type="fed.analytix_log_stats",
                    heartbeat_timeout=0,
                )
            )
            metric_relay_id = job.add_component("metric_relay", component, ctx)
            comp_ids["metric_relay_id"] = metric_relay_id

            component = ExternalConfigurator(
                component_ids=[metric_relay_id],
            )
            comp_ids["config_preparer_id"] = job.add_component("config_preparer", component, ctx)
        else:
            executor = (
                self._executor
                if self._executor
                else self._get_in_process_executor_cls(self._framework)(
                    task_script_path=self._script,
                    task_script_args=self._script_args,
                    params_exchange_format=self._params_exchange_format,
                    params_transfer_type=self._params_transfer_type,
                )
            )
            job.add_executor(executor, tasks=tasks, ctx=ctx)

        job.add_resources(resources=[self._script], ctx=ctx)
        return comp_ids

    def _get_ex_process_executor_cls(self, framework: FrameworkType) -> Type[ClientAPILauncherExecutor]:
        if framework == FrameworkType.PYTORCH:
            from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

            return PTClientAPILauncherExecutor
        elif framework == FrameworkType.TENSORFLOW:
            from nvflare.app_opt.tf.client_api_launcher_executor import TFClientAPILauncherExecutor

            return TFClientAPILauncherExecutor
        else:
            return ClientAPILauncherExecutor

    def _get_in_process_executor_cls(self, framework: FrameworkType) -> Type[InProcessClientAPIExecutor]:
        if framework == FrameworkType.PYTORCH:
            from nvflare.app_opt.pt.in_process_client_api_executor import PTInProcessClientAPIExecutor

            return PTInProcessClientAPIExecutor
        elif framework == FrameworkType.TENSORFLOW:
            from nvflare.app_opt.tf.in_process_client_api_executor import TFInProcessClientAPIExecutor

            return TFInProcessClientAPIExecutor
        else:
            return InProcessClientAPIExecutor


class ScriptRunner(BaseScriptRunner):
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        params_transfer_type: str = TransferType.FULL,
    ):
        """ScriptRunner is used with FedJob API to run or launch a script.

        in-process `launch_external_process=False` uses InProcessClientAPIExecutor (default).
        ex-process `launch_external_process=True` uses ClientAPILauncherExecutor.

        Args:
            script (str): Script to run. For in-process must be a python script path. For ex-process can be any script support by `command`.
            script_args (str): Optional arguments for script (appended to script).
            launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
            command (str): If launch_external_process=True, command to run script (preprended to script). Defaults to "python3".
            framework (str): Framework type to connfigure converter and params exchange formats. Defaults to FrameworkType.PYTORCH.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        """
        super().__init__(
            script=script,
            script_args=script_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=framework,
            params_transfer_type=params_transfer_type,
        )
