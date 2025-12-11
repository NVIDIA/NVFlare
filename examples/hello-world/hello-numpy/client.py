# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
    client side training scripts
"""

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.tracking import SummaryWriter


def train(input_numpy_array: np.ndarray) -> np.ndarray:
    """Mock training the model by adding 1 to the input numpy array.
    In a real application, this would be actual model training.
    """
    return input_numpy_array + 1


def evaluate(input_numpy_array: np.ndarray) -> dict:
    """Mock evaluation the model by returning the mean of the input numpy array.
    In a real application, this would be actual model evaluation.
    """
    return {"weight_mean": np.mean(input_numpy_array)}


def main():
    # Initialize FLARE
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    print(f"Client {client_name} initialized")

    # Initialize summary writer for tracking
    summary_writer = SummaryWriter()

    while flare.is_running():
        # Receive model from server
        input_model = flare.receive()
        print(f"Client {client_name}, current_round={input_model.current_round}")
        print(f"Received weights: {input_model.params}")

        new_params = train(input_model.params[NPConstants.NUMPY_KEY])

        # Evaluate the model
        metrics = evaluate(new_params)
        print(f"Client {client_name} evaluation metrics: {metrics}")

        # Log metrics to summary writer
        global_step = input_model.current_round
        summary_writer.add_scalar(tag="weight_mean", scalar=metrics["weight_mean"], global_step=global_step)

        print(f"Client {client_name} finished training for round {input_model.current_round}")
        print(f"Sending weights: {new_params}")

        # Send updated model back to server
        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: new_params},
            params_type="FULL",
            metrics=metrics,
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
