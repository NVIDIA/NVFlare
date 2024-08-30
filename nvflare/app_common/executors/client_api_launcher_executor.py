# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType, write_config_to_file
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils.attributes_exportable import ExportMode


class ClientAPILauncherExecutor(LauncherExecutor):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
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
        params_exchange_format: str = ExchangeFormat.NUMPY,
        params_transfer_type: str = TransferType.FULL,
        config_file_name: str = CLIENT_API_CONFIG,
    ) -> None:
        """Initializes the ClientAPILauncherExecutor.

        Args:
            pipe_id (str): Identifier for obtaining the Pipe from NVFlare components.
            launcher_id (Optional[str]): Identifier for obtaining the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the Launcher's "launch_task" method to complete (None for no timeout).
            task_wait_timeout (Optional[float]): Timeout for retrieving the task result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process.
                This value should be greater than the time needed for sending the whole result.
            external_pre_init_timeout (float): Time to wait for external process before it calls flare.init().
            peer_read_timeout (float, optional): time to wait for peer to accept sent message.
            monitor_interval (float): Interval for monitoring the launcher.
            read_interval (float): Interval for reading from the pipe.
            heartbeat_interval (float): Interval for sending heartbeat to the peer.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer.
            workers (int): Number of worker threads needed.
            train_with_evaluation (bool): Whether to run training with global model evaluation.
            train_task_name (str): Task name of train mode.
            evaluate_task_name (str): Task name of evaluate mode.
            submit_model_task_name (str): Task name of submit_model mode.
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare executor side to controller side.
            params_exchange_format (str): What format to exchange the parameters.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent.
            config_file_name (str): The config file name to write attributes into, the client api will read in this file.
        """
        LauncherExecutor.__init__(
            self,
            pipe_id=pipe_id,
            launcher_id=launcher_id,
            launch_timeout=launch_timeout,
            task_wait_timeout=task_wait_timeout,
            last_result_transfer_timeout=last_result_transfer_timeout,
            external_pre_init_timeout=external_pre_init_timeout,
            peer_read_timeout=peer_read_timeout,
            monitor_interval=monitor_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            workers=workers,
            train_with_evaluation=train_with_evaluation,
            train_task_name=train_task_name,
            evaluate_task_name=evaluate_task_name,
            submit_model_task_name=submit_model_task_name,
            from_nvflare_converter_id=from_nvflare_converter_id,
            to_nvflare_converter_id=to_nvflare_converter_id,
        )

        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type
        self._config_file_name = config_file_name

    def initialize(self, fl_ctx: FLContext) -> None:
        self.prepare_config_for_launch(fl_ctx)
        super().initialize(fl_ctx)

    def prepare_config_for_launch(self, fl_ctx: FLContext):
        pipe_export_class, pipe_export_args = self.pipe.export(ExportMode.PEER)
        task_exchange_attributes = {
            ConfigKey.TRAIN_WITH_EVAL: self._train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: self._params_exchange_format,
            ConfigKey.TRANSFER_TYPE: self._params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: self._train_task_name,
            ConfigKey.EVAL_TASK_NAME: self._evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: self._submit_model_task_name,
            ConfigKey.PIPE_CHANNEL_NAME: self.get_pipe_channel_name(),
            ConfigKey.PIPE: {
                ConfigKey.CLASS_NAME: pipe_export_class,
                ConfigKey.ARG: pipe_export_args,
            },
            ConfigKey.HEARTBEAT_TIMEOUT: self.heartbeat_timeout,
        }

        config_data = {
            ConfigKey.TASK_EXCHANGE: task_exchange_attributes,
            FLMetaKey.SITE_NAME: fl_ctx.get_identity_name(),
            FLMetaKey.JOB_ID: fl_ctx.get_job_id(),
        }

        config_file_path = self._get_external_config_file_path(fl_ctx)
        write_config_to_file(config_data=config_data, config_file_path=config_file_path)

    def _get_external_config_file_path(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_config_directory = workspace.get_app_config_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(app_config_directory, self._config_file_name)
        return config_file_path
