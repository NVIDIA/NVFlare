# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from abc import ABC, abstractmethod
from typing import List

from tests.integration_test.site_launcher import ServerProperties, SiteProperties


class JobResultValidator(ABC):
    @abstractmethod
    def validate_results(self, server_data: ServerProperties, client_data: List[SiteProperties], run_data) -> bool:
        pass


class FinishJobResultValidator(JobResultValidator):
    def validate_results(self, server_data, client_data, run_data) -> bool:
        # check run folder exist
        server_run_dir = os.path.join(server_data.root_dir, run_data["job_id"])

        if not os.path.exists(server_run_dir):
            print(f"{self.__class__.__name__}: server run dir {server_run_dir} doesn't exist.")
            return False

        for client_prop in client_data:
            client_run_dir = os.path.join(client_prop.root_dir, run_data["job_id"])
            if not os.path.exists(client_run_dir):
                print(f"{self.__class__.__name__}: client run dir {client_run_dir} doesn't exist.")
                return False
