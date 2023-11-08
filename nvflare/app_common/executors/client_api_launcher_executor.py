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
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.data_exchange.constants import ExchangeFormat
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.client.config import ClientConfig, ConfigKey, TransferType
from nvflare.client.constants import CONFIG_EXCHANGE
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.validation_utils import check_object_type


class ClientAPILauncherExecutor(LauncherExecutor):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        wait_timeout: Optional[float] = None,
        result_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 5.0,
        peer_read_timeout: Optional[float] = None,
        result_poll_interval: float = 0.001,
        read_interval: float = 0.001,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        workers: int = 4,
        train_with_evaluation: bool = True,
        train_task_name: str = "train",
        evaluate_task_name: str = "evaluate",
        submit_model_task_name: str = "submit_model",
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        launch_once: bool = True,
        params_exchange_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
    ) -> None:
        """Initializes the ClientAPILauncherExecutor.

        Args:
            pipe_id (Optional[str]): Identifier for obtaining the Pipe from NVFlare components.
            launcher_id (Optional[str]): Identifier for obtaining the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the Launcher's "launch_task" method to complete (None for no timeout).
            wait_timeout (Optional[float]): Timeout for the Launcher's "wait_task" method to complete (None for no timeout).
            result_timeout (Optional[float]): Timeout for retrieving the result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process (default: 5.0).
                This value should be greater than the time needed for sending the whole result.
            peer_read_timeout (Optional[float]): Timeout for waiting the task to be read by the peer from the pipe (None for no timeout).
            result_poll_interval (float): Interval for polling task results from the pipe (default: 0.5).
            read_interval (float): Interval for reading from the pipe (default: 0.5).
            heartbeat_interval (float): Interval for sending heartbeat to the peer (default: 5.0).
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer (default: 30.0).
            workers (int): Number of worker threads needed (default: 4).
            train_with_evaluation (bool): Whether to run training with global model evaluation (default: True).
            train_task_name (str): Task name of train mode (default: train).
            evaluate_task_name (str): Task name of evaluate mode (default: evaluate).
            submit_model_task_name (str): Task name of submit_model mode (default: submit_model).
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare executor side to controller side.
            launch_once (bool): Whether to launch just once for the whole job (default: True). True means only the first task
                will trigger `launcher.launch_task`. Which is efficient when the data setup is taking a lot of time.
            params_exchange_format (ExchangeFormat): What format to exchange the parameters.
            params_transfer_type (TransferType): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent.
        """
        super().__init__(
            pipe_id=pipe_id,
            launcher_id=launcher_id,
            launch_timeout=launch_timeout,
            wait_timeout=wait_timeout,
            result_timeout=result_timeout,
            last_result_transfer_timeout=last_result_transfer_timeout,
            peer_read_timeout=peer_read_timeout,
            result_poll_interval=result_poll_interval,
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
            launch_once=launch_once,
        )

        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type

    def _init_pipe(self, fl_ctx: FLContext) -> None:
        engine = fl_ctx.get_engine()
        pipe_name = f"{fl_ctx.get_identity_name()}-{fl_ctx.get_job_id()}"

        # gets Pipe using _pipe_id
        pipe = engine.get_component(self._pipe_id)
        check_object_type(self._pipe_id, pipe, Pipe)

        if isinstance(pipe, FilePipe):
            if pipe.root_path is None or pipe.root_path == "":
                app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
                pipe.root_path = os.path.abspath(app_dir)
            pipe.mode = Mode.PASSIVE
        else:
            raise RuntimeError(f"Pipe ({self._pipe_id}) of type ({type(pipe)}) is not supported")

        self._pipe = pipe
        self._pipe_name = pipe_name

    def prepare_config_for_launch(self, shareable: Shareable, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_EXCHANGE)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)

        # prepare config exchange for Client API
        client_config = ClientConfig()
        config_dict = client_config.config
        config_dict[ConfigKey.TRAIN_WITH_EVAL] = self._train_with_evaluation
        config_dict[ConfigKey.EXCHANGE_FORMAT] = self._params_exchange_format
        config_dict[ConfigKey.TRANSFER_TYPE] = self._params_transfer_type
        config_dict[ConfigKey.LAUNCH_ONCE] = self._launch_once
        config_dict[ConfigKey.TRAIN_TASK_NAME] = self._train_task_name
        config_dict[ConfigKey.EVAL_TASK_NAME] = self._evaluate_task_name
        config_dict[ConfigKey.SUBMIT_MODEL_TASK_NAME] = self._submit_model_task_name
        config_dict[ConfigKey.PIPE_NAME] = self._pipe_name
        config_dict[ConfigKey.PIPE_CLASS] = self._pipe.__class__.__name__
        config_dict[ConfigKey.PIPE_ARGS] = self._get_external_pipe_args()
        config_dict[ConfigKey.TOTAL_ROUNDS] = total_rounds
        config_dict[ConfigKey.SITE_NAME] = fl_ctx.get_identity_name()
        config_dict[ConfigKey.JOB_ID] = fl_ctx.get_job_id()
        client_config.to_json(config_file)

    def _get_external_pipe_args(self) -> dict:
        args = {}
        if isinstance(self._pipe, FilePipe):
            args["root_path"] = self._pipe.root_path
            args["mode"] = Mode.ACTIVE
        return args
