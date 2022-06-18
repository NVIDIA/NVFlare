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

from .job_result_validator import FinishJobResultValidator

TB_PATH = "tb_events"


class TBResultValidator(FinishJobResultValidator):
    def validate_finished_results(self, job_result, client_props) -> bool:
        server_run_dir = job_result["workspace_root"]
        server_tb_root_dir = os.path.join(server_run_dir, TB_PATH)

        if not os.path.exists(server_tb_root_dir):
            self.logger.info("server_tb_root_dir {server_tb_root_dir} doesn't exist.")
            return False

        for client_prop in client_props:
            client_run_dir = os.path.join(client_prop.root_dir, job_result["job_id"])
            client_side_client_tb_dir = os.path.join(client_run_dir, TB_PATH, client_prop.name)
            if not os.path.exists(client_side_client_tb_dir):
                self.logger.info("client_side_client_tb_dir {client_side_client_tb_dir} doesn't exist.")
                return False

        return True
