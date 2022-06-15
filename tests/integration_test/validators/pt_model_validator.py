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

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.app_common.app_constant import DefaultCheckpointFileName

from .job_result_validator import FinishJobResultValidator


class PTModelValidator(FinishJobResultValidator):
    def validate_finished_results(self, job_result, client_props) -> bool:
        server_run_dir = job_result["workspace_root"]

        server_models_dir = os.path.join(server_run_dir, WorkspaceConstants.APP_PREFIX + "server")
        if not os.path.exists(server_models_dir):
            self.logger.info(f"models dir {server_models_dir} doesn't exist.")
            return False

        model_path = os.path.join(server_models_dir, DefaultCheckpointFileName.GLOBAL_MODEL)
        if not os.path.isfile(model_path):
            self.logger.info(f"model_path {model_path} doesn't exist.")
            return False

        return True
