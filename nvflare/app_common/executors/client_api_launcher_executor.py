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
from typing import Dict, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors.file_pipe_launcher_executor import FilePipeLauncherExecutor
from nvflare.app_common.model_exchange.constants import ModelExchangeFormat
from nvflare.client.config import ClientConfig, ConfigKey, TransferType
from nvflare.client.constants import CONFIG_EXCHANGE


class ClientAPILauncherExecutor(FilePipeLauncherExecutor):
    def __init__(
        self,
        data_exchange_path: Optional[str] = None,
        pipe_id: Optional[str] = None,
        pipe_name: str = "pipe",
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        task_wait_time: Optional[float] = None,
        task_read_wait_time: Optional[float] = None,
        result_poll_interval: float = 0.1,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        workers: int = 1,
        training: bool = True,
        global_evaluation: bool = True,
        params_exchange_format: ModelExchangeFormat = ModelExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
    ) -> None:
        """Initializes the ClientAPILauncherExecutor.

        Args:
            data_exchange_path (Optional[str]): Path used for data exchange. If None, the "app_dir" of the running job will be used.
                If pipe_id is provided, will use the Pipe gets from pipe_id.
            pipe_id (Optional[str]): Identifier used to get the Pipe from NVFlare components.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            launcher_id (Optional[str]): Identifier used to get the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the "launch" method to end. None means never timeout.
            task_wait_time (Optional[float]): Time to wait for tasks to complete before exiting the executor. None means never timeout.
            task_read_wait_time (Optional[float]): Time to wait for task results from the pipe. None means no wait.
            result_poll_interval (float): Interval for polling task results from the pipe. Defaults to 0.1.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
            workers (int): Number of worker threads needed.
            training (bool): Whether to run training using global model. Defaults to True.
            global_evaluation (bool): Whether to run evaluation on global model. Defaults to True.
            params_exchange_format (ModelExchangeFormat): What format to exchange the parameters.
            params_transfer_type (TransferType): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent.
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare executor side to controller side.
        """
        super().__init__(
            data_exchange_path=data_exchange_path,
            pipe_id=pipe_id,
            pipe_name=pipe_name,
            launcher_id=launcher_id,
            launch_timeout=launch_timeout,
            task_wait_time=task_wait_time,
            task_read_wait_time=task_read_wait_time,
            result_poll_interval=result_poll_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            workers=workers,
            training=training,
            global_evaluation=global_evaluation,
            from_nvflare_converter_id=from_nvflare_converter_id,
            to_nvflare_converter_id=to_nvflare_converter_id,
        )

        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type

    def prepare_config_for_launch(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_EXCHANGE)

        client_config = ClientConfig()
        self._update_config_exchange_dict(client_config.config)
        client_config.to_json(config_file)

    def _update_config_exchange_dict(self, config: Dict):
        config[ConfigKey.GLOBAL_EVAL] = self._global_evaluation
        config[ConfigKey.TRAINING] = self._training
        config[ConfigKey.EXCHANGE_FORMAT] = self._params_exchange_format
        config[ConfigKey.EXCHANGE_PATH] = self._data_exchange_path
        config[ConfigKey.TRANSFER_TYPE] = self._params_transfer_type
