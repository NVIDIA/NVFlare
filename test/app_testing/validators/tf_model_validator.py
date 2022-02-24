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
from test.app_testing.app_result_validator import AppResultValidator


def check_tf_results(server_data, client_data, run_data):
    run_number = run_data["run_number"]
    server_dir = server_data["server_path"]

    server_run_dir = os.path.join(server_dir, "run_" + str(run_number))

    if not os.path.exists(server_run_dir):
        print(f"check_tf_results: server run dir {server_run_dir} doesn't exist.")
        return False

    models_dir = os.path.join(server_run_dir, "app_server")
    if not os.path.exists(models_dir):
        print(f"check_sag_results: models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "tf2weights.pickle")
    if not os.path.isfile(model_path):
        print(f"check_tf_results: model_path {model_path} doesn't exist.")
        return False

    try:
        data = pickle.load(open(model_path, "rb"))
        print(f"check_tf_result: Data loaded: {data}.")
        assert "weights" in data
        assert "meta" in data
    except Exception as e:
        print(f"Exception in validating TF model: {e.__str__()}")
        return False

    return True


class TFModelValidator(AppResultValidator):
    def __init__(self):
        super().__init__()

    def validate_results(self, server_data, client_data, run_data) -> bool:
        return check_tf_results(server_data, client_data, run_data)
