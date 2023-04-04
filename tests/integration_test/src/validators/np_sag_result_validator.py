# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from .job_result_validator import FinishJobResultValidator


class NumpySAGResultValidator(FinishJobResultValidator):
    def __init__(self, expected_result):
        super().__init__()
        self.expected_result = np.array(expected_result)

    def validate_finished_results(self, job_result, client_props) -> bool:
        server_run_dir = job_result["workspace_root"]

        models_dir = os.path.join(server_run_dir, "models")
        if not os.path.exists(models_dir):
            self.logger.error(f"models dir {models_dir} doesn't exist.")
            return False

        model_path = os.path.join(models_dir, "server.npy")
        if not os.path.isfile(model_path):
            self.logger.error(f"model_path {model_path} doesn't exist.")
            return False

        try:
            data = np.load(model_path)
            self.logger.info(f"data loaded: {data}.")
            np.testing.assert_equal(data, self.expected_result)
        except Exception as e:
            self.logger.error(f"exception happens: {e.__str__()}")
            return False

        return True
