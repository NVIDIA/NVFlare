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

from typing import Type

from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.import_utils import optional_import


class FrameworkType:
    RAW = "raw"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class ScriptRunner:
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
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
        """
        self._script = script
        self._script_args = script_args
        self._command = command
        self._launch_external_process = launch_external_process
        self._framework = framework

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

    def add_to_fed_job(self, job, ctx, **kwargs):
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
            from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
            from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
            from nvflare.app_common.widgets.metric_relay import MetricRelay
            from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

            component = CellPipe(
                mode="PASSIVE",
                site_name="{SITE_NAME}",
                token="{JOB_ID}",
                root_url="{ROOT_URL}",
                secure_mode="{SECURE_MODE}",
                workspace_dir="{WORKSPACE}",
            )
            pipe_id = job.add_component("pipe", component, ctx)
            comp_ids["pipe_id"] = pipe_id

            component = SubprocessLauncher(
                script=self._command + " custom/" + self._script + " " + self._script_args,
            )
            launcher_id = job.add_component("launcher", component, ctx)
            comp_ids["launcher_id"] = launcher_id

            executor = self._get_ex_process_executor_cls(self._framework)(
                pipe_id=pipe_id,
                launcher_id=launcher_id,
                params_exchange_format=self._params_exchange_format,
                heartbeat_timeout=0,
            )
            job.add_executor(executor, tasks=tasks, ctx=ctx)

            component = CellPipe(
                mode="PASSIVE",
                site_name="{SITE_NAME}",
                token="{JOB_ID}",
                root_url="{ROOT_URL}",
                secure_mode="{SECURE_MODE}",
                workspace_dir="{WORKSPACE}",
            )
            metric_pipe_id = job.add_component("metrics_pipe", component, ctx)
            comp_ids["metric_pipe_id"] = metric_pipe_id

            component = MetricRelay(
                pipe_id=metric_pipe_id,
                event_type="fed.analytix_log_stats",
                heartbeat_timeout=0,
            )
            metric_relay_id = job.add_component("metric_relay", component, ctx)
            comp_ids["metric_relay_id"] = metric_relay_id

            component = ExternalConfigurator(
                component_ids=[metric_relay_id],
            )
            comp_ids["config_preparer_id"] = job.add_component("config_preparer", component, ctx)
        else:
            executor = self._get_in_process_executor_cls(self._framework)(
                task_script_path=self._script,
                task_script_args=self._script_args,
                params_exchange_format=self._params_exchange_format,
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
