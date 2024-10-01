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

from typing import List, Union

from bionemo_constants import BioNeMoConstants
from bionemo_inference_processor import BioNeMoInferenceProcessor

from nvflare.app_common.workflows.broadcast_and_process import BroadcastAndProcess


class BioNeMoInference(BroadcastAndProcess):
    def __init__(
        self,
        base_config_path: str = "/workspace/bionemo/examples/protein/esm1nv/conf/base_config.yaml",
        infer_config_path: str = "config/infer.yaml",
        task_name: str = BioNeMoConstants.TASK_INFERENCE,
        min_responses_required: int = 0,
        wait_time_after_min_received: int = 0,
        task_timeout: int = 0,
        clients: Union[List[str], None] = None,
    ):
        """A controller for running federated BiNeMo model inference on the clients.

        Args:
            config_path: BioNeMo inference config file.
            task_name: name of the task to be sent to clients to share configs and model weight.
            min_responses_required: min number of responses required. 0 means all clients.
            wait_time_after_min_received: how long (secs) to wait after min responses are received
            task_timeout: max amount of time to wait for the task to end. 0 means never time out.
            clients: names of the clients to send config. Defaults to `None`.
                If `None`, the task will be sent to all clients.
                If list of client names, the config will be only be sent to the listed clients.
        """

        if clients is not None:
            if not isinstance(clients, list):
                raise ValueError(f"Expected list of client names but received {clients}")

        BroadcastAndProcess.__init__(
            self,
            processor=BioNeMoInferenceProcessor(base_config_path=base_config_path, infer_config_path=infer_config_path),
            task_name=task_name,
            min_responses_required=min_responses_required,
            wait_time_after_min_received=wait_time_after_min_received,
            timeout=task_timeout,
            clients=clients,
        )
