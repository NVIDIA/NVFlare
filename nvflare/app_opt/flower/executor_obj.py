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
from typing import List, Union

from nvflare.job_config.fed_object import FedObject

from .defs import Constant
from .executor import FlowerExecutor


class FlowerExecutorObj(FlowerExecutor, FedObject):
    def __init__(
        self,
        client_app: str = "client:app",
        client_files: Union[str, List[str]] = None,
        start_task_name=Constant.START_TASK_NAME,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        client_shutdown_timeout=5.0,
    ):
        super().__init__(
            client_app=client_app,
            client_files=client_files,
            start_task_name=start_task_name,
            configure_task_name=configure_task_name,
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
            client_shutdown_timeout=client_shutdown_timeout,
        )
        self.client_files = client_files

    def get_resources(self):
        return self.client_files
