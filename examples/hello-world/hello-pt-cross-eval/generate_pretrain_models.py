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
Generate pre-trained PyTorch models for cross-site evaluation demo.

In CSE, the server provides models to clients for evaluation.
Clients validate these models on their local data.
"""

import os

import torch
from model import SimpleNetwork

SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


def _save_model(model, model_dir: str, model_file: str):
    """Save PyTorch model state dict to file."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    # Create pre-trained models for server and clients
    # In practice, these would be models trained on different datasets or with different hyperparameters

    print("Generating server pre-trained models...")
    # Server Model 1: Randomly initialized
    model_1 = SimpleNetwork()
    _save_model(model=model_1, model_dir=SERVER_MODEL_DIR, model_file="server_1.pt")

    # Server Model 2: Xavier initialization
    model_2 = SimpleNetwork()
    for param in model_2.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    _save_model(model=model_2, model_dir=SERVER_MODEL_DIR, model_file="server_2.pt")

    print("\nGenerating client pre-trained models...")
    # Client models - one for each site
    for site_id in [1, 2]:
        model = SimpleNetwork()
        # Apply different initialization per site
        if site_id == 1:
            torch.nn.init.normal_(model.conv1.weight, mean=0.0, std=0.01)
        else:
            torch.nn.init.kaiming_normal_(model.conv1.weight)

        _save_model(model=model, model_dir=CLIENT_MODEL_DIR, model_file=f"site-{site_id}.pt")

    print("\nPre-trained models generated:")
    print(f"  Server models: {SERVER_MODEL_DIR}")
    print(f"  Client models: {CLIENT_MODEL_DIR}")
    print("You can now run cross-site evaluation with these models.")
