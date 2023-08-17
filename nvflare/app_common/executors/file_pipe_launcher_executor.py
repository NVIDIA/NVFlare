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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.client.config import ConfigKey
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.fuel.utils.validation_utils import check_object_type


class FilePipeLauncherExecutor(LauncherExecutor):
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
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
    ) -> None:
        """Initializes the FilePipeLauncherExecutor.

        Args:
            data_exchange_path (Optional[str]): Path used for data exchange. If None, "app_dir" will be used.
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
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare executor side to controller side.
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
        )

        self._data_exchange_path = data_exchange_path

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)

        engine = fl_ctx.get_engine()

        # gets pipe
        if self._pipe_id:
            pipe: FilePipe = engine.get_component(self._pipe_id)
            check_object_type(self._pipe_id, pipe, FilePipe)
            self._data_exchange_path = pipe.root_path
        else:
            # gets data_exchange_path
            if self._data_exchange_path is None:
                app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
                self._data_exchange_path = app_dir
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
        self._update_config_exchange(fl_ctx)

    def _update_config_exchange_dict(self, config: dict):
        super()._update_config_exchange_dict(config)
        config[ConfigKey.EXCHANGE_PATH] = os.path.abspath(self._data_exchange_path)
