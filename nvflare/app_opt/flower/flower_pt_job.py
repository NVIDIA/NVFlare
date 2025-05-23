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

from typing import List, Optional

from nvflare.app_common.tie.defs import Constant
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

from .flower_job import FlowerJob


class FlowerPyTorchJob(FlowerJob):
    def __init__(
        self,
        name: str,
        flower_content: str,
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        database: str = "",
        superlink_ready_timeout: float = 10.0,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        client_shutdown_timeout=5.0,
        stream_metrics=False,
        analytics_receiver=None,
        extra_env: dict = None,
    ):
        """
        Flower Job.

        Args:
            name (str): Name of the job.
            flower_content (str): Content for the flower job.
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
            stream_metrics (bool, optional): Whether to stream metrics from Flower client to Flare
            analytics_receiver (AnalyticsReceiver, optional): the AnalyticsReceiver to use to process received metrics.
            extra_env (dict, optional): optional extra env variables to be passed to Flower client
        """
        analytics_receiver = (
            analytics_receiver if analytics_receiver else TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
        )

        super().__init__(
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
            stream_metrics=stream_metrics,
            analytics_receiver=analytics_receiver,
            extra_env=extra_env,
        )
