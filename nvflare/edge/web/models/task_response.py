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
from nvflare.edge.web.models.base_model import BaseModel


class TaskResponse(BaseModel):
    def __init__(
        self,
        status: str,
        job_id: str = None,
        retry_wait: int = None,
        task_id: str = None,
        task_name: str = None,
        task_data: dict = None,
        cookie: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.status = status
        self.job_id = job_id
        self.retry_wait = retry_wait
        self.task_id = task_id
        self.task_name = task_name
        self.task_data = task_data
        self.cookie = cookie

        if kwargs:
            self.update(kwargs)
