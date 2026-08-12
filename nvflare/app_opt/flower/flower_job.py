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

import os.path
from typing import List, Optional

from nvflare.app_common.metrics_exchange.metrics_sender import ANALYTICS_BOOTSTRAP_ENV
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.app_opt.flower.defs import Constant as FlowerConstant
from nvflare.job_config.api import FedJob

from .controller import FlowerController
from .executor import FlowerExecutor
from .path_utils import validate_flower_app_path


class FlowerJob(FedJob):
    def __init__(
        self,
        name: str,
        flower_content: Optional[str] = None,
        flower_app_path: Optional[str] = None,
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        database: str = "",
        superlink_ready_timeout: float = 10.0,
        configure_task_timeout=FlowerConstant.CONFIG_TASK_TIMEOUT,
        start_task_timeout=FlowerConstant.START_TASK_TIMEOUT,
        max_client_op_interval: float = FlowerConstant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = FlowerConstant.WORKFLOW_PROGRESS_TIMEOUT,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        client_shutdown_timeout=5.0,
        extra_env: Optional[dict] = None,
        run_config: Optional[dict] = None,
        allow_runtime_dependency_installation: bool = False,
    ):
        """
        Flower Job.

        Args:
            name (str): Name of the job.
            flower_content (str, optional): Local directory path containing Flower app code (BYOC mode).
            flower_app_path (str, optional): Absolute path to pre-deployed Flower app on the server (pre-deployed mode). The server distributes the app to clients via Flower's FAB mechanism.
            min_clients (int, optional): The minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): List of mandatory clients for the job. Defaults to None.
            database (str, optional): Database string. Defaults to "".
            superlink_ready_timeout (float, optional): Timeout for the superlink to be ready. Defaults to 10.0 seconds.
            configure_task_timeout (float, optional): Timeout for configuring the task. Defaults to Constant.CONFIG_TASK_TIMEOUT.
            start_task_timeout (float, optional): Timeout for starting the task. Defaults to Constant.START_TASK_TIMEOUT.
            max_client_op_interval (float, optional): Maximum interval between client operations. Defaults to Constant.MAX_CLIENT_OP_INTERVAL.
            progress_timeout (float, optional): Timeout for workflow progress. Defaults to Constant.WORKFLOW_PROGRESS_TIMEOUT.
            per_msg_timeout (float, optional): Timeout for receiving individual messages. Defaults to 10.0 seconds.
            tx_timeout (float, optional): Timeout for transmitting data. Defaults to 100.0 seconds.
            client_shutdown_timeout (float, optional): Timeout for client shutdown. Defaults to 5.0 seconds.
            extra_env (dict, optional): optional extra env variables to be passed to Flower client
            run_config (dict, optional): optional dict for flwr run --run-config arguments
            allow_runtime_dependency_installation (bool, optional): whether to allow dynamic dependency installation. Defaults to False. (only flwr>=1.29)
        """
        if flower_content and flower_app_path:
            raise ValueError("Specify either 'flower_content' (BYOC) or 'flower_app_path' (pre-deployed), not both.")
        if not flower_content and not flower_app_path:
            raise ValueError("One of 'flower_content' or 'flower_app_path' must be provided.")

        if flower_content:
            if not os.path.isdir(flower_content):
                raise ValueError(f"{flower_content} is not a valid directory")

        flower_env = extra_env.copy() if extra_env is not None else None
        if flower_env is not None and ANALYTICS_BOOTSTRAP_ENV in flower_env:
            raise ValueError(
                f"extra_env key {ANALYTICS_BOOTSTRAP_ENV!r} is reserved; FlowerJob passes its analytics bootstrap"
            )

        # Validate flower_app_path format and security
        if flower_app_path:
            validate_flower_app_path(flower_app_path)

        # Mark pre-deployed jobs in meta.json.
        extra_meta = {}
        if flower_app_path:
            extra_meta[FlowerConstant.FLOWER_PREDEPLOYED] = True

        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            meta_props=extra_meta if extra_meta else None,
        )

        controller = FlowerController(
            database=database,
            superlink_ready_timeout=superlink_ready_timeout,
            configure_task_timeout=configure_task_timeout,
            start_task_timeout=start_task_timeout,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
            run_config=run_config,
            allow_runtime_dependency_installation=allow_runtime_dependency_installation,
            flower_app_path=flower_app_path,
        )
        self.to_server(controller)
        if flower_content:
            self.to_server(obj=flower_content)

        self.to_clients(MetricRelay(), "metric_relay")
        executor = FlowerExecutor(
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
            client_shutdown_timeout=client_shutdown_timeout,
            extra_env=flower_env,
            allow_runtime_dependency_installation=allow_runtime_dependency_installation,
        )
        self.to_clients(executor)
        if flower_content:
            self.to_clients(obj=flower_content)
