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
from abc import ABC
from typing import List

from nvflare.app_common.app_constant import AppConstants

from .job_result_validator import FinishJobResultValidator


class BaseCrossValResultValidator(FinishJobResultValidator, ABC):
    def __init__(self, server_model_names: List[str]):
        super().__init__()
        self.server_model_names = server_model_names

    def check_cross_validation_result(self, job_result, client_props, n_clients=-1, global_model_eval=False):
        client_names = []
        client_len = len(client_props) if n_clients == -1 else n_clients
        for i in range(client_len):
            client_names.append(client_props[i].name)

        server_run_dir = job_result["workspace_root"]
        cross_val_dir = os.path.join(server_run_dir, AppConstants.CROSS_VAL_DIR)
        if not os.path.exists(cross_val_dir):
            self.logger.info(f"models dir {cross_val_dir} doesn't exist.")
            return False

        model_shareable_dir = os.path.join(cross_val_dir, AppConstants.CROSS_VAL_MODEL_DIR_NAME)
        if not os.path.exists(model_shareable_dir):
            self.logger.info(f"model shareable directory {model_shareable_dir} doesn't exist.")
            return False

        result_shareable_dir = os.path.join(cross_val_dir, AppConstants.CROSS_VAL_RESULTS_DIR_NAME)
        if not os.path.exists(result_shareable_dir):
            self.logger.info(f"result shareable directory {result_shareable_dir} doesn't exist.")
            return False

        # There should be three files in model_shareable
        server_model_names = [f"SRV_{i}" for i in self.server_model_names]
        model_file_names = server_model_names.copy()
        if not global_model_eval:
            model_file_names = model_file_names + client_names
        self.logger.info(f"Model files to look for: {model_file_names}")

        for model_file_name in model_file_names:
            model_file = os.path.join(model_shareable_dir, model_file_name)
            if not os.path.exists(model_file):
                self.logger.info(f"model {model_file} doesn't exist in model shareable directory.")
                return False

        # Check all the results
        # results_file_names = ["client_1_server", "client_0_server", "client_1_client_0", "client_1_client_1",
        #                       "client_0_client_1", "client_0_client_0"]
        results_file_names = [f"{x}_{y}" for x in client_names for y in server_model_names]
        if not global_model_eval:
            for client_name in client_names:
                results_file_names += [f"{client_name}_{x}" for x in client_names]
        self.logger.info(f"Result files to look for: {results_file_names}")

        for results_file_name in results_file_names:
            result_file = os.path.join(result_shareable_dir, results_file_name)
            if not os.path.exists(result_file):
                self.logger.info(f"result {result_file} doesn't exist in result shareable directory.")
                return False

        return True


class GlobalModelEvalValidator(BaseCrossValResultValidator):
    def validate_finished_results(self, job_result, client_props) -> bool:
        return self.check_cross_validation_result(job_result, client_props, n_clients=-1, global_model_eval=True)


class CrossValResultValidator(BaseCrossValResultValidator):
    def validate_finished_results(self, job_result, client_props) -> bool:
        return self.check_cross_validation_result(job_result, client_props)


class CrossValSingleClientResultValidator(BaseCrossValResultValidator):
    def validate_finished_results(self, job_result, client_props) -> bool:
        return self.check_cross_validation_result(job_result, client_props, n_clients=1)
