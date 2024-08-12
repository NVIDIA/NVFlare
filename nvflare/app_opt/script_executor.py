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
from typing import List, Union

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.external_configurator import ExternalConfigurator
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.job_config.fed_job import FedJob
from nvflare.job_config.job_object import ExecutorJobObj

torch, torch_ok = optional_import(module="torch")
if torch_ok:
    from nvflare.app_opt.pt.params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter

    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.PYTORCH
else:
    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.NUMPY

tensorflow, tf_ok = optional_import(module="tensorflow")
if tf_ok:
    from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter


class ScriptExecutor(ExecutorJobObj):
    def __init__(
        self,
        script: str,
        script_args: str = "",
        resources: Union[str, List[str]] = None,
        launch_external_process: bool = False,
        launch_once: bool = True,
        params_transfer_type: TransferType = TransferType.FULL,
        params_exchange_format=DEFAULT_PARAMS_EXCHANGE_FORMAT,
        tasks: List[str] = None,
        gpu: Union[int, List[int]] = None,
    ):
        """ScriptExecutor is used with FedJob API to run or launch a script.

        in-process `launch_external_process=False`  (default)
            - uses InProcessClientAPIExecutor (default)
            - adds script to as external resource

        ex-process `launch_external_process=True` 
            - uses ClientAPILauncherExecutor, SubprocessLauncher, and related components
            - adds any `custom/xxx` as external resources

        Args:
            script (str): Script to run. For in-process must be a python script path. For ex-process can be any command supported by python subprocess.
            script_args (str): Optional arguments for script.
            launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
            launch_once (bool): If True launch script once, else launch for every task. Defaults to True.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent. Defaults to TransferType.FULL.
            params_exchange_format (str): Format to exchange the parameters. Defaults based on detected imports.
            tasks: List of tasks the executor should handle. Defaults to `None`. If `None`, all tasks will be handled using `[*]`.
            gpu: GPU index or list of GPU indices used for simulating the run on that target.
        """
        self.script = script
        self.script_args = script_args
        self.resources = resources
        self.launch_external_process = launch_external_process
        self.launch_once = launch_once
        self.params_transfer_type = params_transfer_type
        self.params_exchange_format = params_exchange_format

        self.from_nvflare_converter = None
        self.to_nvflare_converter = None

        if launch_external_process:
            executor = ClientAPILauncherExecutor(
                pipe_id="pipe",
                launcher_id="launcher",
                params_exchange_format=self.params_exchange_format,
                params_transfer_type=self.params_transfer_type,
                from_nvflare_converter_id="from_nvflare",
                to_nvflare_converter_id="to_nvflare",
            )
        else:
            executor = InProcessClientAPIExecutor(
                task_script_path=os.path.basename(self.script),
                task_script_args=self.script_args,
                params_exchange_format=self.params_exchange_format,
                params_transfer_type=self.params_transfer_type,
                from_nvflare_converter_id="from_nvflare",
                to_nvflare_converter_id="to_nvflare",
            )

        if self.launch_external_process:
            self.resources = resources
        else:
            self.resources = self.script

        super().__init__(
            executor=executor,
            resources=self.resources,
            tasks=tasks,
            gpu=gpu,
        )

        if torch_ok:
            if params_exchange_format == ExchangeFormat.PYTORCH:
                self.from_nvflare_converter = NumpyToPTParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
                self.to_nvflare_converter = PTToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
        if tf_ok:
            if params_exchange_format == ExchangeFormat.NUMPY:
                self.from_nvflare_converter = NumpyToKerasModelParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
                self.to_nvflare_converter = KerasModelToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )

    def add_to_job(self, job: FedJob, target: str):
        super().add_to_job(job, target)
        if self.launch_external_process:
            component = SubprocessLauncher(
                script=self.script + " " + self.script_args,
                launch_once=self.launch_once,
            )
            job.add_object(obj=component, target=target, id="launcher")

            component = CellPipe(
                mode="PASSIVE",
                site_name="{SITE_NAME}",
                token="{JOB_ID}",
                root_url="{ROOT_URL}",
                secure_mode="{SECURE_MODE}",
                workspace_dir="{WORKSPACE}",
            )
            job.add_object(obj=component, target=target, id="pipe")

            component = CellPipe(
                mode="PASSIVE",
                site_name="{SITE_NAME}",
                token="{JOB_ID}",
                root_url="{ROOT_URL}",
                secure_mode="{SECURE_MODE}",
                workspace_dir="{WORKSPACE}",
            )
            job.add_object(obj=component, target=target, id="metrics_pipe")

            component = MetricRelay(
                pipe_id="metrics_pipe",
                event_type="fed.analytix_log_stats",
            )
            job.add_object(obj=component, target=target, id="metric_relay")

            component = ExternalConfigurator(
                component_ids=["metric_relay"],
            )
            job.add_object(obj=component, target=target, id="config_preparer")

        if self.from_nvflare_converter:
            job.add_object(obj=self.from_nvflare_converter, target=target, id="from_nvflare")
        if self.to_nvflare_converter:
            job.add_object(obj=self.to_nvflare_converter, target=target, id="to_nvflare")
