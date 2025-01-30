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
import torch

def compute_loss_over_dataset(
    model: torch.nn.Module | None = None,
    loss: torch.nn.modules.loss._Loss | None = None,
    dataloader: torch.utils.data.DataLoader | None = None,
) -> float:
    """
    Compute the average loss over a dataset.

    Args:
        model: The model to use for predictions.
        loss: The loss function to use.
        dataloader: The dataloader for the dataset.

    Returns:
        The average loss over the dataset.
    """
    # Check if all required arguments are provided
    if model is None or loss is None or dataloader is None:
        raise ValueError("All arguments (model, loss, dataloader) must be provided.")

    epoch_loss = 0
    total_samples = 0
    with torch.no_grad():
        # Iterate over the dataloader
        for x, y in dataloader:
            # Make predictions
            pred = model(x)
            # Compute the loss
            ls = loss(pred, y)
            # Accumulate the loss and total samples
            epoch_loss += ls.item()  # Get the scalar value of the loss
            total_samples += 1

    # Check if there are any samples in the dataset
    if total_samples == 0:
        raise ValueError("The dataset is empty.")

    # Return the average loss
    return epoch_loss / total_samples
