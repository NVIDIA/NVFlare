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
from typing import Optional

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.security.logging import secure_format_traceback


class CmdTaskController(ModelController):
    def __init__(
        self,
        task_name: str = "cmd_task",
        task_data: Optional[dict] = None,
        task_meta: Optional[dict] = None,
        num_clients: Optional[int] = None,
        min_responses: Optional[int] = None,
        timeout: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.task_data = task_data if task_data is not None else {"task_name": task_name}
        self.task_meta = task_meta if task_meta is not None else {"status": "request"}
        self.num_clients = num_clients
        self.min_responses = min_responses
        self.timeout = timeout

    def run(self):
        self.info(f"{self.task_name} task started.")

        try:
            task = FLModel(params=self.task_data, meta=self.task_meta, current_round=0, total_rounds=1)
            clients = self.sample_clients(self.num_clients)
            if self.min_responses is not None and self.min_responses > len(clients):
                raise RuntimeError(
                    f"min_responses={self.min_responses} exceeds sampled clients={len(clients)}; "
                    "either lower min_responses, increase num_clients, or set a non-zero timeout."
                )
            self.send_task_and_wait(
                task_name=self.task_name,
                targets=clients,
                data=task,
                min_responses=self.min_responses,
                timeout=self.timeout,
            )
            self.info(f"Finished {self.task_name}.")
        except Exception as ex:
            msg = secure_format_traceback()
            self.panic(f"task {self.task_name} failed with exception {msg}")
            raise ex

    def send_task_and_wait(self, task_name, targets, data, min_responses=None, timeout=0):
        return self.send_model_and_wait(
            task_name=task_name, targets=targets, data=data, min_responses=min_responses, timeout=timeout
        )
