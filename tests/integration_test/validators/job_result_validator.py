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

import logging
import os
from abc import ABC, abstractmethod
from typing import List

from tests.integration_test.site_launcher import SiteProperties


class JobResultValidator(ABC):
    @abstractmethod
    def validate_results(self, job_result, client_props: List[SiteProperties]) -> bool:
        pass


class FinishJobResultValidator(JobResultValidator, ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate_finished_results(self, job_result, client_props) -> bool:
        pass

    def validate_results(self, job_result, client_props) -> bool:
        # check run folder exist
        server_run_dir = job_result["workspace_root"]

        if not os.path.exists(server_run_dir):
            self.logger.info(f"server run dir {server_run_dir} doesn't exist.")
            return False

        for client_prop in client_props:
            client_run_dir = os.path.join(client_prop.root_dir, job_result["job_id"])
            if not os.path.exists(client_run_dir):
                self.logger.info(f"client run dir {client_run_dir} doesn't exist.")
                return False

        if not self.validate_finished_results(job_result, client_props):
            return False
        return True
