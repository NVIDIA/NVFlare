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
from typing import Dict, List

from nvflare.apis.executor import Executor
from nvflare.app_common.job.base_app import BaseApp
from nvflare.private.fed.client.client_json_config import _ExecutorDef


class ClientApp(BaseApp):
    def __init__(self) -> None:
        super().__init__()

        self.executors: [_ExecutorDef] = []

    def add_executor(self, tasks: List[str], executor: Executor):
        if not isinstance(executor, Executor):
            raise RuntimeError(f"workflow must be type of Executor, but got {executor.__class__}")

        e = _ExecutorDef()
        e.tasks = tasks
        e.executor = executor
        self.executors.append(e)

