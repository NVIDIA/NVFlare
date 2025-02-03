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
    device: torch.device | None = None,
) -> float:
    """
    Compute the average loss over a dataset.

    Args:
        model: The model to use for predictions.
        loss: The loss function to use.
        dataloader: The dataloader for the dataset.
        device: The device to use for computation.

    Returns:
        The average loss over the dataset.
    """
    # Check if all required arguments are provided
    if model is None or loss is None or dataloader is None:
        raise ValueError("All arguments (model, loss, dataloader) must be provided.")

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        # Iterate over the dataloader
        for x, y in dataloader:
            # Move data to the specified device
            x, y = x.to(device), y.to(device)
            # Make predictions
            pred = model(x)
            # Compute the loss
            ls = loss(pred, y)
            # Accumulate the loss
            epoch_loss += ls.item() * x.size(0)
    # Return the average loss
    return epoch_loss / len(dataloader.dataset)
