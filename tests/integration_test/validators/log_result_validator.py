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
import re

from .job_result_validator import FinishJobResultValidator


class LogResultValidator(FinishJobResultValidator):
    def __init__(self, expected_in_result=None, expected_not_in_result=None):
        super().__init__()
        self.expected_in_result = expected_in_result
        self.expected_not_in_result = expected_not_in_result

    def validate_finished_results(self, job_result, client_props) -> bool:
        server_run_dir = job_result["workspace_root"]

        log_txt = os.path.join(server_run_dir, "log.txt")
        if not os.path.exists(log_txt):
            self.logger.info(f"log file {log_txt} doesn't exist.")
            return False

        try:
            with open(log_txt) as f:
                server_log = f.read()
            if expected_in_result:
                assert re.search(expected_in_result, server_log)
            if expected_not_in_result:
                assert not re.search(expected_not_in_result, server_log)
        except Exception as e:
            self.logger.info(f"exception happens: {e.__str__()}")
            return False

        return True
