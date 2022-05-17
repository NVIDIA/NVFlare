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
import pickle

from nvflare.apis.fl_constant import WorkspaceConstants

from .job_result_validator import FinishJobResultValidator


def _check_tf_results(server_data, run_data):
    server_run_dir = os.path.join(server_data.root_dir, run_data["job_id"])
    server_models_dir = os.path.join(server_run_dir, WorkspaceConstants.APP_PREFIX + "server")
    if not os.path.exists(server_models_dir):
        print(f"_check_tf_results: models dir {server_models_dir} doesn't exist.")
        return False

    model_path = os.path.join(server_models_dir, "tf2weights.pickle")
    if not os.path.isfile(model_path):
        print(f"_check_tf_results: model_path {model_path} doesn't exist.")
        return False

    try:
        data = pickle.load(open(model_path, "rb"))
        print(f"_check_tf_results: Data loaded: {data}.")
        assert "weights" in data
        assert "meta" in data
    except Exception as e:
        print(f"Exception in validating TF model: {e.__str__()}")
        return False

    return True


class TFModelValidator(FinishJobResultValidator):
    def validate_results(self, server_data, client_data, run_data) -> bool:
        super().validate_results(server_data, client_data, run_data)
        return _check_tf_results(server_data, run_data)
