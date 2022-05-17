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
    def validate_results(self, server_data, client_data, run_data) -> bool:
        super().validate_results(server_data, client_data, run_data)
        server_run_dir = os.path.join(server_data.root_dir, run_data["job_id"])
        server_tb_root_dir = os.path.join(server_run_dir, TB_PATH)

        if not os.path.exists(server_tb_root_dir):
            print(f"{self.__class__.__name__}: server_tb_root_dir {server_tb_root_dir} doesn't exist.")
            return False

        for client_prop in client_data:
            client_run_dir = os.path.join(client_prop.root_dir, run_data["job_id"])
            client_side_client_tb_dir = os.path.join(client_run_dir, TB_PATH, client_prop.name)
            if not os.path.exists(client_side_client_tb_dir):
                print(
                    f"{self.__class__.__name__}: client_side_client_tb_dir {client_side_client_tb_dir} doesn't exist."
                )
                return False

        return True
