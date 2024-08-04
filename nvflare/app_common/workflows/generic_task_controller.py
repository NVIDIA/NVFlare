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


from nvflare.app_common.abstract.generic_task import GenericTask
from nvflare.app_common.workflows.model_controller import ModelController


class GenericTaskController(ModelController):
    def __init__(self, task_name: str = "gen_task", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name

    def run(self):
        self.info(f"{self.task_name} task started.")

        # use FLModel structure, add empty model
        task = GenericTask()
        clients = self.sample_clients()
        results = self.send_task_and_wait(task_name=self.task_name, targets=clients, data=task)

        self.info("Finished etl.")

    def send_task_and_wait(self, task_name, targets, data):
        return self.send_model_and_wait(task_name=task_name, targets=targets, data=data)
