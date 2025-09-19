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
from nvflare.client.tracking import SummaryWriter


class SimpleNumpyModel:
    """
    A simple numpy-based model for demonstration purposes.
    This model represents a basic neural network with weights that can be trained
    using federated learning.
    """

    def __init__(self):
        """
        Initialize the model with fixed weights for demonstration.
        This is a simple 3x3 weight matrix for the hello-world example.
        """
        # Initialize weights with the standard starting values for this example
        self.weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def get_weights(self):
        """Get the current model weights."""
        return self.weights.copy()

    def set_weights(self, weights):
        """Set the model weights."""
        self.weights = weights.copy()

    def train_step(self, learning_rate=1.0):
        """
        Perform one training step by adding a delta to the weights.
        This is a simplified training step for demonstration purposes.

        Args:
            learning_rate: Learning rate for the training step

        Returns:
            The updated weights
        """
        # Simple training: add 1 to each weight (simulating gradient descent)
        self.weights = self.weights + learning_rate
        return self.weights

    def evaluate(self):
        """
        Evaluate the model and return metrics.

        Returns:
            Dictionary containing evaluation metrics
        """
        # Simple evaluation: return the mean of the weights as a mock metric
        # In a real application, this would be actual model performance on test data
        weight_mean = np.mean(self.weights)
        return {"weight_mean": weight_mean}


def main():
    # Initialize the model
    model = SimpleNumpyModel()

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

        # Load the received model weights
        if input_model.params == {}:
            # Initialize with default weights if this is the first round
            params = model.get_weights()
        else:
            params = np.array(input_model.params["numpy_key"], dtype=np.float32)

        model.set_weights(params)

        # Perform local training
        print(f"Client {client_name} starting training...")
        new_params = model.train_step(learning_rate=1.0)

        # Evaluate the model
        metrics = model.evaluate()
        print(f"Client {client_name} evaluation metrics: {metrics}")

        # Log metrics to summary writer
        global_step = input_model.current_round
        summary_writer.add_scalar(tag="weight_mean", scalar=metrics["weight_mean"], global_step=global_step)

        print(f"Client {client_name} finished training for round {input_model.current_round}")
        print(f"Sending weights: {new_params}")

        # Send updated model back to server
        output_model = flare.FLModel(
            params={"numpy_key": new_params},
            params_type="FULL",
            metrics=metrics,
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
