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
from test.app_testing.app_result_validator import AppResultValidator

TB_PATH = "tb_events"


class TBResultValidator(AppResultValidator):
    def validate_results(self, server_data, client_data, run_data) -> bool:
        run_number = run_data["run_number"]
        server_path = server_data["server_path"]

        server_run_dir = os.path.join(server_path, f"run_{run_number}")
        server_tb_root_dir = os.path.join(server_run_dir, TB_PATH)
        if not os.path.exists(server_tb_root_dir):
            print(f"tb validate results: server_tb_root_dir {server_tb_root_dir} doesn't exist.")
            return False

        for i, client_path in enumerate(client_data["client_paths"]):
            client_run_dir = os.path.join(client_path, f"run_{run_number}")
            client_side_client_tb_dir = os.path.join(client_run_dir, TB_PATH, client_data["client_names"][i])
            if not os.path.exists(client_side_client_tb_dir):
                print(f"tb validate results: client_side_client_tb_dir {client_side_client_tb_dir} doesn't exist.")
                return False

        return True
