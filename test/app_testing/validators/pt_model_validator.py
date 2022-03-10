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
from collections import OrderedDict
from test.app_testing.app_result_validator import AppResultValidator

import numpy as np


def check_pt_results(server_data, client_data, run_data):
    run_number = run_data["run_number"]
    server_dir = server_data["server_path"]

    server_run_dir = os.path.join(server_dir, "run_" + str(run_number))
    intended_model = OrderedDict(
        [
            (
                "model",
                OrderedDict(
                    [("linear.weight", np.array([[-0.1234, -0.1452, -0.1570]])), ("linear.bias", np.array([0.4643]))]
                ),
            ),
            ("train_conf", {"train": {"model": "SimpleNetwork"}}),
        ]
    )

    if not os.path.exists(server_run_dir):
        print(f"check_sag_results: server run dir {server_run_dir} doesn't exist.")
        return False

    models_dir = os.path.join(server_run_dir, "app_server")
    if not os.path.exists(models_dir):
        print(f"check_sag_results: models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "FL_global_model.pt")
    if not os.path.isfile(model_path):
        print(f"check_sag_results: model_path {model_path} doesn't exist.")
        return False

    return True


class PTModelValidator(AppResultValidator):
    def __init__(self):
        super(PTModelValidator, self).__init__()

    def validate_results(self, server_data, client_data, run_data) -> bool:
        return check_pt_results(server_data, client_data, run_data)
