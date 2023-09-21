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
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.app_common.model_exchange.constants import ModelExchangeFormat
from nvflare.client.config import ClientConfig, ConfigKey, TransferType
from nvflare.client.constants import CONFIG_EXCHANGE
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.fuel.utils.validation_utils import check_object_type


class ClientAPILauncherExecutor(LauncherExecutor):
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
        launch_once: bool = False,
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
            launch_once (bool): Whether to launch just once for the whole. Default is True, means only the first task
                will trigger `launcher.launch_task`. Which is efficient when the data setup is taking a lot of time.
        """
        super().__init__(
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
            launch_once=launch_once,
        )

        self._data_exchange_path = data_exchange_path
        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)

        engine = fl_ctx.get_engine()

        # gets FilePipe using _pipe_id or initialize a new one
        if self._pipe_id:
            pipe: FilePipe = engine.get_component(self._pipe_id)
            check_object_type(self._pipe_id, pipe, FilePipe)
            self._data_exchange_path = pipe.root_path
        else:
            # gets data_exchange_path
            if self._data_exchange_path is None or self._data_exchange_path == "":
                app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
                self._data_exchange_path = os.path.abspath(app_dir)
            elif not os.path.isabs(self._data_exchange_path):
                raise RuntimeError("data exchange path needs to be absolute.")
            pipe = FilePipe(mode=Mode.ACTIVE, root_path=self._data_exchange_path)

        # init pipe
        flare_decomposers.register()
        common_decomposers.register()
        pipe.open(self._pipe_name)
        self.pipe_handler = PipeHandler(
            pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        self.pipe_handler.start()

    def prepare_config_for_launch(self, shareable: Shareable, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_EXCHANGE)

        # prepare config exchange for Client API
        client_config = ClientConfig()
        self._update_config_exchange_dict(client_config.config)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        client_config.config[ConfigKey.TOTAL_ROUNDS] = total_rounds
        client_config.config[ConfigKey.SITE_NAME] = fl_ctx.get_identity_name()
        client_config.config[ConfigKey.JOB_ID] = fl_ctx.get_job_id()
        client_config.to_json(config_file)

    def _update_config_exchange_dict(self, config: Dict):
        config[ConfigKey.GLOBAL_EVAL] = self._global_evaluation
        config[ConfigKey.TRAINING] = self._training
        config[ConfigKey.EXCHANGE_FORMAT] = self._params_exchange_format
        config[ConfigKey.EXCHANGE_PATH] = self._data_exchange_path
        config[ConfigKey.TRANSFER_TYPE] = self._params_transfer_type
        config[ConfigKey.LAUNCH_ONCE] = self._launch_once
