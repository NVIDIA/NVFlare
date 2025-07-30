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
from abc import ABC, abstractmethod
from typing import Any

from nvflare.apis.filter import Filter
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType


class ExecEnv(ABC):

    @abstractmethod
    def deploy(self, job: FedJob):
        pass


class Recipe(ABC):

    def __init__(self, job: FedJob):
        self.job = job

    def add_client_data_filter(self, filter: Filter, tasks=None):
        self.job.to_clients(filter, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_client_result_filter(self, filter: Filter, tasks=None):
        self.job.to_clients(filter, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def add_server_data_filter(self, filter: Filter, tasks=None):
        self.job.to_server(filter, filter_type=FilterType.TASK_DATA, tasks=tasks)

    def add_server_result_filter(self, filter: Filter, tasks=None):
        self.job.to_server(filter, filter_type=FilterType.TASK_RESULT, tasks=tasks)

    def export(self, job_dir: str):
        self.job.export_job(job_dir)

    def execute(self, env: ExecEnv) -> Any:
        return env.deploy(self.job)
