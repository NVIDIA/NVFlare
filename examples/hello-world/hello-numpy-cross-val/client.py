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

"""Simple NumPy training script for cross-site evaluation example.

This script performs mock training for demonstrating the training+CSE workflow.
"""

import argparse

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def train(input_numpy_array: np.ndarray) -> np.ndarray:
    """Mock training by adding 1 to the input numpy array.

    In a real application, this would be actual model training.
    """
    return input_numpy_array + 1


def evaluate(input_numpy_array: np.ndarray) -> dict:
    """Mock evaluation by returning the mean of the input numpy array.

    In a real application, this would be actual model evaluation.
    """
    return {"weight_mean": float(np.mean(input_numpy_array))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    args = parser.parse_args()

    # Initialize FLARE
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    print(f"Client {client_name} initialized")

    # Track last trained params so we can submit them when CSE asks for our local model.
    last_params = None

    while flare.is_running():
        input_model = flare.receive()
        print(f"Client {client_name}, current_round={input_model.current_round}")

        if flare.is_train():
            # Training task: receive global model, train, send update.
            if input_model.params is None or NPConstants.NUMPY_KEY not in input_model.params:
                raise RuntimeError(
                    "Train task received no model params (params is None or missing numpy_key). "
                    "Server requires a valid initial model; empty response would break aggregation."
                )
            input_np_arr = input_model.params[NPConstants.NUMPY_KEY]
            print(f"Received weights: {input_np_arr}")
            new_params = train(input_np_arr)
            last_params = new_params
            metrics = evaluate(new_params)
            print(f"Client {client_name} evaluation metrics: {metrics}")
            print(f"Client {client_name} finished training for round {input_model.current_round}")
            if args.update_type == "diff":
                params_to_send = new_params - input_np_arr
                params_type = flare.ParamsType.DIFF
            else:
                params_to_send = new_params
                params_type = flare.ParamsType.FULL
            print(f"Sending weights: {params_to_send}")
            flare.send(
                flare.FLModel(
                    params={NPConstants.NUMPY_KEY: params_to_send},
                    params_type=params_type,
                    metrics=metrics,
                    current_round=input_model.current_round,
                )
            )

        elif flare.is_evaluate():
            # Validate task: evaluate the received model and send metrics only (no params).
            if input_model.params is None or NPConstants.NUMPY_KEY not in input_model.params:
                flare.send(flare.FLModel(metrics={}))
                continue
            input_np_arr = input_model.params[NPConstants.NUMPY_KEY]
            metrics = evaluate(input_np_arr)
            print(f"Client {client_name} validation metrics: {metrics}")
            flare.send(flare.FLModel(metrics=metrics))

        elif flare.is_submit_model():
            # Submit local model for cross-site evaluation (must be WEIGHTS DXO).
            if last_params is None:
                raise RuntimeError(
                    "submit_model called but no local model (last_params) available. "
                    "CSE expects client weights; run training first or fix job order."
                )
            print(f"Client {client_name} submitting local model")
            flare.send(
                flare.FLModel(
                    params={NPConstants.NUMPY_KEY: last_params},
                    params_type=flare.ParamsType.FULL,
                )
            )

        else:
            # Unknown task; send empty metrics so the protocol is satisfied.
            flare.send(flare.FLModel(metrics={}))


if __name__ == "__main__":
    main()
