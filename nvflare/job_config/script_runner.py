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

from typing import Optional, Type

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.import_utils import optional_import


class FrameworkType:
    RAW = "raw"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class SubprocessLauncherArgs:
    def __init__(self, launch_once: bool = True, clean_up_script: Optional[str] = None):
        self.launch_once = launch_once
        self.clean_up_script = clean_up_script


class MetricRelayArgs:
    def __init__(
        self,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=60.0,
        pipe_channel_name=PipeChannelName.METRIC,
        event_type: str = "fed." + ANALYTIC_EVENT_TYPE,
        fed_event: bool = True,
    ) -> None:
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.pipe_channel_name = pipe_channel_name
        self.event_type = event_type
        self.fed_event = fed_event


class ClientAPILauncherExecutorArgs:
    def __init__(
        self,
        launch_timeout: Optional[float] = None,
        task_wait_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 300.0,
        external_pre_init_timeout: float = 60.0,
        peer_read_timeout: Optional[float] = 60.0,
        monitor_interval: float = 0.01,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 60.0,
        workers: int = 4,
        train_with_evaluation: bool = True,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        config_file_name: str = CLIENT_API_CONFIG,
    ) -> None:

        self.launch_timeout = launch_timeout
        self.task_wait_timeout = task_wait_timeout
        self.last_result_transfer_timeout = last_result_transfer_timeout
        self.external_pre_init_timeout = external_pre_init_timeout
        self.peer_read_timeout = peer_read_timeout
        self.monitor_interval = monitor_interval
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.workers = workers
        self.train_with_evaluation = train_with_evaluation
        self.train_task_name = train_task_name
        self.evaluate_task_name = evaluate_task_name
        self.submit_model_task_name = submit_model_task_name
        self.from_nvflare_converter_id = from_nvflare_converter_id
        self.to_nvflare_converter_id = to_nvflare_converter_id
        self.config_file_name = config_file_name


class InProcessClientAPIExecutorArgs:
    def __init__(
        self,
        task_wait_time: Optional[float] = None,
        result_pull_interval: float = 0.5,
        log_pull_interval: Optional[float] = None,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        train_with_evaluation: bool = True,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ) -> None:
        self.task_wait_time = task_wait_time
        self.result_pull_interval = result_pull_interval
        self.log_pull_interval = log_pull_interval
        self.train_with_evaluation = train_with_evaluation
        self.train_task_name = train_task_name
        self.evaluate_task_name = evaluate_task_name
        self.submit_model_task_name = submit_model_task_name
        self.from_nvflare_converter_id = from_nvflare_converter_id
        self.to_nvflare_converter_id = to_nvflare_converter_id


class ScriptRunner:
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        params_transfer_type: str = TransferType.FULL,
        subprocess_launcher_args: Optional[SubprocessLauncherArgs] = None,
        metric_relay_args: Optional[MetricRelayArgs] = None,
        client_api_launcher_executor_args: Optional[ClientAPILauncherExecutorArgs] = None,
        in_process_client_api_executor_args: Optional[InProcessClientAPIExecutorArgs] = None,
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
            subprocess_launcher_args: arguments for SubprocessLauncher
            metric_relay_args: arguments for MetricRelay
            client_api_launcher_executor_args: arguments for ClientAPILauncherExecutor
            in_process_client_api_executor_args arguments for InProcessClientAPIExecutor
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

        self.subprocess_launcher_args = (
            subprocess_launcher_args if subprocess_launcher_args else SubprocessLauncherArgs()
        )
        self.metric_relay_args = metric_relay_args if metric_relay_args else MetricRelayArgs()
        self.client_api_launcher_executor_args = (
            client_api_launcher_executor_args if client_api_launcher_executor_args else ClientAPILauncherExecutorArgs()
        )
        self.in_process_client_api_executor_args = (
            in_process_client_api_executor_args
            if in_process_client_api_executor_args
            else InProcessClientAPIExecutorArgs()
        )

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
                launch_once=self.subprocess_launcher_args.launch_once,
                clean_up_script=self.subprocess_launcher_args.clean_up_script,
            )
            launcher_id = job.add_component("launcher", component, ctx)
            comp_ids["launcher_id"] = launcher_id

            executor = self._get_ex_process_executor_cls(self._framework)(
                pipe_id=pipe_id,
                launcher_id=launcher_id,
                launch_timeout=self.client_api_launcher_executor_args.launch_timeout,
                task_wait_timeout=self.client_api_launcher_executor_args.task_wait_timeout,
                last_result_transfer_timeout=self.client_api_launcher_executor_args.last_result_transfer_timeout,
                external_pre_init_timeout=self.client_api_launcher_executor_args.external_pre_init_timeout,
                peer_read_timeout=self.client_api_launcher_executor_args.peer_read_timeout,
                monitor_interval=self.client_api_launcher_executor_args.monitor_interval,
                read_interval=self.client_api_launcher_executor_args.read_interval,
                heartbeat_interval=self.client_api_launcher_executor_args.heartbeat_interval,
                heartbeat_timeout=self.client_api_launcher_executor_args.heartbeat_timeout,
                workers=self.client_api_launcher_executor_args.workers,
                train_with_evaluation=self.client_api_launcher_executor_args.train_with_evaluation,
                train_task_name=self.client_api_launcher_executor_args.train_task_name,
                evaluate_task_name=self.client_api_launcher_executor_args.evaluate_task_name,
                submit_model_task_name=self.client_api_launcher_executor_args.submit_model_task_name,
                from_nvflare_converter_id=self.client_api_launcher_executor_args.from_nvflare_converter_id,
                to_nvflare_converter_id=self.client_api_launcher_executor_args.to_nvflare_converter_id,
                params_exchange_format=self._params_exchange_format,
                params_transfer_type=self._params_transfer_type,
                config_file_name=self.client_api_launcher_executor_args.config_file_name,
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
                event_type=self.metric_relay_args.event_type,
                heartbeat_timeout=self.metric_relay_args.heartbeat_timeout,
                heartbeat_interval=self.metric_relay_args.heartbeat_interval,
                read_interval=self.metric_relay_args.read_interval,
                pipe_channel_name=self.metric_relay_args.pipe_channel_name,
                fed_event=self.metric_relay_args.fed_event,
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
                task_wait_time=self.in_process_client_api_executor_args.task_wait_time,
                result_pull_interval=self.in_process_client_api_executor_args.result_pull_interval,
                log_pull_interval=self.in_process_client_api_executor_args.log_pull_interval,
                params_exchange_format=self._params_exchange_format,
                params_transfer_type=self._params_transfer_type,
                from_nvflare_converter_id=self.in_process_client_api_executor_args.from_nvflare_converter_id,
                to_nvflare_converter_id=self.in_process_client_api_executor_args.to_nvflare_converter_id,
                train_with_evaluation=self.in_process_client_api_executor_args.train_with_evaluation,
                train_task_name=self.in_process_client_api_executor_args.train_task_name,
                evaluate_task_name=self.in_process_client_api_executor_args.evaluate_task_name,
                submit_model_task_name=self.in_process_client_api_executor_args.submit_model_task_name,
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
