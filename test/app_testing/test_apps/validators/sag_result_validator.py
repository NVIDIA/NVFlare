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

import numpy as np


def check_sag_results(server_data, client_data, run_data):
    run_number = run_data["run_number"]
    server_dir = server_data["server_path"]

    server_run_dir = os.path.join(server_dir, "run_" + str(run_number))
    intended_model = np.array(
        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype="float32",
    )

    if not os.path.exists(server_run_dir):
        print(f"check_sag_results: server run dir {server_run_dir} doesn't exist.")
        return False

    models_dir = os.path.join(server_run_dir, "models")
    if not os.path.exists(models_dir):
        print(f"check_sag_results: models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "server.npy")
    if not os.path.isfile(model_path):
        print(f"check_sag_results: model_path {model_path} doesn't exist.")
        return False

    try:
        data = np.load(model_path)
        print(f"check_sag_result: Data loaded: {data}.")
        np.testing.assert_equal(data, intended_model)
    except Exception as e:
        print(f"Exception in validating ScatterAndGather model: {e.__str__()}")
        return False

    return True


class SAGResultValidator(AppResultValidator):
    def __init__(self):
        super(SAGResultValidator, self).__init__()

    def validate_results(self, server_data, client_data, run_data) -> bool:
        # pass
        # print(f"CrossValValidator server data: {server_data}")
        # print(f"CrossValValidator client data: {client_data}")
        # print(f"CrossValValidator run data: {run_data}")

        fed_avg_result = check_sag_results(server_data, client_data, run_data)

        print(f"ScatterAndGather Result: {fed_avg_result}")

        if not fed_avg_result:
            raise ValueError("Scatter and gather failed.")

        return fed_avg_result
