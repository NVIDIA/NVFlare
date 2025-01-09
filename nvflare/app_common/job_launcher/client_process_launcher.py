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

from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.utils.job_launcher_utils import generate_client_command


class ClientProcessJobLauncher(ProcessJobLauncher):
    def get_command(self, job_meta, fl_ctx) -> str:
        return generate_client_command(fl_ctx)
