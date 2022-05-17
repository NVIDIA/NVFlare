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

import numpy as np

from .job_result_validator import FinishJobResultValidator


def _print_info(msg: str):
    print(f"_check_np_sag_results: {msg}")


def _check_np_sag_results(server_data, run_data, expected_result: np.array):
    server_run_dir = os.path.join(server_data.root_dir, run_data["job_id"])

    models_dir = os.path.join(server_run_dir, "models")
    if not os.path.exists(models_dir):
        _print_info(f"models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "server.npy")
    if not os.path.isfile(model_path):
        _print_info(f"model_path {model_path} doesn't exist.")
        return False

    try:
        data = np.load(model_path)
        _print_info(f"data loaded: {data}.")
        np.testing.assert_equal(data, expected_result)
    except Exception as e:
        _print_info(f"exception happens: {e.__str__()}")
        return False

    return True


class NumpySAGResultValidator(FinishJobResultValidator):
    def __init__(self, expected_result):
        self.expected_result = np.array(expected_result)

    def validate_results(self, server_data, client_data, run_data) -> bool:
        super().validate_results(server_data, client_data, run_data)
        return _check_np_sag_results(server_data, run_data, self.expected_result)
