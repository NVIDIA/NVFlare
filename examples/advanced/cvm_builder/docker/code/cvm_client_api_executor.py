# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Measured CVM executor with no job-configurable command or script path."""

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode


class CVMClientAPIExecutor(ClientAPIExecutor):
    def __init__(self, params_exchange_format="pytorch", server_expected_format="numpy"):
        super().__init__(
            execution_mode=ExecutionMode.IN_PROCESS,
            task_script_path="/local/custom/client.py",
            params_exchange_format=params_exchange_format,
            server_expected_format=server_expected_format,
        )
