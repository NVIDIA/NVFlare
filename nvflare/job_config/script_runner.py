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

from typing import Optional, Union

from nvflare.apis.fl_constant import ExchangeFormat, SystemVarName
from nvflare.app_common.abstract.launcher import Launcher
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.app_common.filters.params_converter_filter import ParamsConverterFilter
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe, Mode
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.validation_utils import check_str
from nvflare.job_config.defs import FilterType

from .api import FedJob, validate_object_for_job


class FrameworkType:
    RAW = "raw"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class PipeConnectType:
    VIA_ROOT = "via_root"
    VIA_CP = "via_cp"
    VIA_RELAY = "via_relay"


_PIPE_CONNECT_URL = {
    PipeConnectType.VIA_CP: "{" + SystemVarName.CP_URL + "}",
    PipeConnectType.VIA_RELAY: "{" + SystemVarName.RELAY_URL + "}",
    PipeConnectType.VIA_ROOT: "{" + SystemVarName.ROOT_URL + "}",
}


def _add_pt_pt_filter(job, ctx):
    from nvflare.app_opt.pt.tensor_params_converter import PTReceiveParamsConverter, PTSendParamsConverter

    job.add_component("pt_send", PTSendParamsConverter(), ctx)
    job.add_component("pt_receive", PTReceiveParamsConverter(), ctx)
    job.add_filter(
        ParamsConverterFilter(params_converter_id="pt_receive"),
        FilterType.TASK_DATA,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    job.add_filter(
        ParamsConverterFilter(params_converter_id="pt_send"),
        FilterType.TASK_RESULT,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    return


def _add_np_pt_filter(job, ctx):
    from nvflare.app_opt.pt.numpy_params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter

    job.add_component("np_to_pt", NumpyToPTParamsConverter(), ctx)
    job.add_component("pt_to_np", PTToNumpyParamsConverter(), ctx)
    job.add_filter(
        ParamsConverterFilter(params_converter_id="np_to_pt"),
        FilterType.TASK_DATA,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    job.add_filter(
        ParamsConverterFilter(params_converter_id="pt_to_np"),
        FilterType.TASK_RESULT,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    return


def _add_np_keras_filter(job, ctx):
    from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter

    job.add_component("keras_to_np", KerasModelToNumpyParamsConverter(), ctx)
    job.add_component("np_to_keras", NumpyToKerasModelParamsConverter(), ctx)
    job.add_filter(
        ParamsConverterFilter(params_converter_id="np_to_keras"),
        FilterType.TASK_DATA,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    job.add_filter(
        ParamsConverterFilter(params_converter_id="keras_to_np"),
        FilterType.TASK_RESULT,
        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION, AppConstants.TASK_SUBMIT_MODEL],
        ctx,
    )
    return


# ScriptRunner supported builtin exchange format combinations
# other combinations users need to make sure the params exchange
# between nvflare server <-> nvflare client and nvflare client <-> script is good
AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS = {
    (ExchangeFormat.PYTORCH, ExchangeFormat.PYTORCH): _add_pt_pt_filter,
    (ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH): _add_np_pt_filter,
    (ExchangeFormat.NUMPY, ExchangeFormat.KERAS_LAYER_WEIGHTS): _add_np_keras_filter,
}


class BaseScriptRunner:
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        executor: Union[ClientAPILauncherExecutor, InProcessClientAPIExecutor, None] = None,
        task_pipe: Optional[Pipe] = None,
        launcher: Optional[Launcher] = None,
        metric_relay: Optional[MetricRelay] = None,
        metric_pipe: Optional[Pipe] = None,
        pipe_connect_type: str = None,
        server_expected_format: str = ExchangeFormat.NUMPY,
        script_expected_format: str = ExchangeFormat.NUMPY,
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
            command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
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

            pipe_connect_type: how pipe peers are to be connected:
                Via Root: peers are both connected to the root of the cellnet
                Via Relay: peers are both connected to the relay if a relay is used; otherwise via root.
                Via CP: peers are both connected to the CP
                If not specified, will be via CP.
        """
        self._script = script
        self._script_args = script_args
        self._command = command
        self._launch_external_process = launch_external_process
        self._server_expected_format = server_expected_format
        self._script_expected_format = script_expected_format
        self._pipe_connect_type = pipe_connect_type

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

        if pipe_connect_type:
            check_str("pipe_connect_type", pipe_connect_type)
            valid_connect_types = [PipeConnectType.VIA_CP, PipeConnectType.VIA_RELAY, PipeConnectType.VIA_RELAY]
            if pipe_connect_type not in valid_connect_types:
                raise ValueError(f"invalid pipe_connect_type '{pipe_connect_type}': must be {valid_connect_types}")

        self._metric_pipe = metric_pipe
        self._metric_relay = metric_relay
        self._task_pipe = task_pipe
        self._executor = executor
        self._launcher = launcher

    def _create_cell_pipe(self):
        ct = self._pipe_connect_type
        if not ct:
            ct = PipeConnectType.VIA_CP
        conn_url = _PIPE_CONNECT_URL.get(ct)
        if not conn_url:
            raise RuntimeError(f"cannot determine pipe connect url for {self._pipe_connect_type}")

        return CellPipe(
            mode=Mode.PASSIVE,
            site_name="{" + SystemVarName.SITE_NAME + "}",
            token="{" + SystemVarName.JOB_ID + "}",
            root_url=conn_url,
            secure_mode="{" + SystemVarName.SECURE_MODE + "}",
            workspace_dir="{" + SystemVarName.WORKSPACE + "}",
        )

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
            task_pipe = self._task_pipe if self._task_pipe else self._create_cell_pipe()
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
                else ClientAPILauncherExecutor(
                    pipe_id=task_pipe_id,
                    launcher_id=launcher_id,
                    script_expected_format=self._script_expected_format,
                )
            )
            job.add_executor(executor, tasks=tasks, ctx=ctx)

            metric_pipe = self._metric_pipe if self._metric_pipe else self._create_cell_pipe()
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
                else InProcessClientAPIExecutor(
                    task_script_path=self._script,
                    task_script_args=self._script_args,
                    script_expected_format=self._script_expected_format,
                )
            )
            job.add_executor(executor, tasks=tasks, ctx=ctx)

        exchange_format_combination = (
            self._server_expected_format,
            self._script_expected_format,
        )
        if exchange_format_combination in AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS:
            add_filter_cb = AUTO_REGISTERED_EXCHANGE_FORMAT_COMBINATIONS[exchange_format_combination]
            add_filter_cb(job=job, ctx=ctx)
        job.add_resources(resources=[self._script], ctx=ctx)
        return comp_ids


class ScriptRunner(BaseScriptRunner):
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        pipe_connect_type: str = PipeConnectType.VIA_CP,
        server_expected_format: str = ExchangeFormat.NUMPY,
        script_expected_format: str = ExchangeFormat.NUMPY,
    ):
        """ScriptRunner is used with FedJob API to run or launch a script.

        in-process `launch_external_process=False` uses InProcessClientAPIExecutor (default).
        ex-process `launch_external_process=True` uses ClientAPILauncherExecutor.

        It will call "[command][script][script_args]".

        Args:
            script (str): Script to run. For in-process must be a python script path. For ex-process can be any script support by `command`.
            script_args (str): Optional arguments for script (appended to script).
            launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
            command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
            pipe_connect_type (str): how pipe peers are to be connected
        """
        super().__init__(
            script=script,
            script_args=script_args,
            launch_external_process=launch_external_process,
            command=command,
            pipe_connect_type=pipe_connect_type,
            server_expected_format=server_expected_format,
            script_expected_format=script_expected_format,
        )
