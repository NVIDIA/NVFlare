# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nvflare.app_common.tie.defs import Constant
from nvflare.app_opt.flower.flower_job import FlowerJob
from nvflare.client.api import ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.recipe.spec import Recipe


class FlowerRecipe(Recipe):
    """Recipe class for Flower federated learning using NVFlare.

    This class provides a high-level interface for configuring Flower
    federated learning jobs. It wraps the FlowerJob and provides
    a recipe-based interface for easier job configuration and execution.

    Enables metric streaming and use of client API by default.

    Example usage:
        ```python
        recipe = FlowerRecipe(
            name="my_flower_job",
            flower_content="/path/to/flower/content",
            min_clients=2,
            stream_metrics=True
        )
        ```

    Args:
        flower_content (str): Content for the flower job. Required.
        name (str): Name of the job. Defaults to "flower_job".
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
        extra_env (dict, optional): optional extra env variables to be passed to Flower client.
    """

    def __init__(
        self,
        flower_content: str,
        name: str = "flower_job",
        min_clients: int = 1,
        mandatory_clients: Optional[list[str]] = None,
        database: str = "",
        superlink_ready_timeout: float = 10.0,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        client_shutdown_timeout=5.0,
        extra_env: dict = None,
    ):
        """Initialize the FlowerRecipe.

        Creates a FlowerJob and wraps it in the Recipe interface.
        """

        # needs to init client api to stream metrics
        # only external client api works with the current flower integration
        env = {CLIENT_API_TYPE_KEY: ClientAPIType.EX_PROCESS_API.value}

        job = FlowerJob(
            name=name,
            flower_content=flower_content,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            database=database,
            superlink_ready_timeout=superlink_ready_timeout,
            configure_task_timeout=configure_task_timeout,
            start_task_timeout=start_task_timeout,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
            client_shutdown_timeout=client_shutdown_timeout,
            extra_env=env,
        )

        super().__init__(job)
