# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.response_processors.global_weights_initializer import GlobalWeightsInitializer, WeightMethod

from .broadcast_and_process import BroadcastAndProcess


class InitializeGlobalWeights(BroadcastAndProcess):
    def __init__(
        self,
        task_name: str = AppConstants.TASK_GET_WEIGHTS,
        min_responses_required: int = 0,
        wait_time_after_min_received: int = 0,
        task_timeout: int = 0,
        weights_prop_name=AppConstants.GLOBAL_MODEL,
        weight_method: str = WeightMethod.FIRST,
        weights_client_name: Union[str, List[str], None] = None,
    ):
        """A controller for initializing global model weights based on reported weights from clients.

        Args:
            task_name: name of the task to be sent to clients to collect their model weights
            min_responses_required: min number of responses required. 0 means all clients.
            wait_time_after_min_received: how long (secs) to wait after min responses are received
            task_timeout: max amount of time to wait for the task to end. 0 means never time out.
            weights_prop_name: name of the FL Context property to store the global weights
            weight_method: method for determining global model weights. Defaults to `WeightMethod.FIRST`.
            weights_client_name: name of the client if the method is "client". Defaults to None.
                If `None`, the task will be sent to all clients (to be used with `weight_method=WeightMethod.FIRST`).
                If list of client names, the task will be only be sent to the listed clients.
        """

        if isinstance(weights_client_name, str):
            clients = [weights_client_name]
        elif isinstance(weights_client_name, list):
            clients = weights_client_name
        else:
            clients = None

        BroadcastAndProcess.__init__(
            self,
            processor=GlobalWeightsInitializer(
                weights_prop_name=weights_prop_name, weight_method=weight_method, client_name=weights_client_name
            ),
            task_name=task_name,
            min_responses_required=min_responses_required,
            wait_time_after_min_received=wait_time_after_min_received,
            timeout=task_timeout,
            clients=clients,
        )
