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
"""Minimal NumPy client used to validate K8s job-level monitoring."""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def train(input_numpy_array: np.ndarray) -> np.ndarray:
    return input_numpy_array + 1


def evaluate(input_numpy_array: np.ndarray) -> dict:
    return {"weight_mean": float(np.mean(input_numpy_array))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    args = parser.parse_args()

    flare.init()

    while flare.is_running():
        input_model = flare.receive()
        input_np_arr = input_model.params[NPConstants.NUMPY_KEY]

        new_params = train(input_np_arr)
        metrics = evaluate(new_params)

        if args.update_type == "diff":
            params_to_send = new_params - input_np_arr
            params_type = flare.ParamsType.DIFF
        else:
            params_to_send = new_params
            params_type = flare.ParamsType.FULL

        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: params_to_send},
            params_type=params_type,
            metrics=metrics,
            current_round=input_model.current_round,
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
