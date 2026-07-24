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

"""Split learning on MNIST with the Collab API.

The client owns the images and bottom model. The server owns the labels and top
model. Each step sends activations to the server and gradients back to the
client, expressed as ordinary Python function calls.

Run from the ``examples`` directory:

    python -m collab.simple_split_learning.simple_split_learning
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import default_collate
from torchvision import datasets, transforms

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

BATCH_SIZE = 64
HIDDEN_DIM = 256
LEARNING_RATE = 0.05
NUM_STEPS = 200

_bottom_model = None
_bottom_optimizer = None
_client_data = None
_activations = None


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST(root="./data", train=True, download=True, transform=transform)


def get_batch(dataset, step):
    num_batches = math.ceil(len(dataset) / BATCH_SIZE)
    batch_index = step % num_batches
    start = batch_index * BATCH_SIZE
    stop = min(start + BATCH_SIZE, len(dataset))
    return default_collate([dataset[index] for index in range(start, stop)])


@collab.publish
def forward(step):
    """Run the client-side bottom model and return cut-layer activations."""
    global _activations, _bottom_model, _bottom_optimizer, _client_data
    if _bottom_model is None:
        _client_data = load_mnist()
        _bottom_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, HIDDEN_DIM), nn.ReLU())
        _bottom_optimizer = optim.SGD(_bottom_model.parameters(), lr=LEARNING_RATE)

    images, _ = get_batch(_client_data, step)
    _activations = _bottom_model(images)
    return _activations.detach()


@collab.publish
def backward(gradients):
    """Apply server-provided cut-layer gradients to the bottom model."""
    _bottom_optimizer.zero_grad(set_to_none=True)
    _activations.backward(gradients)
    _bottom_optimizer.step()


@collab.main
def split_learning():
    """Run split learning from the server."""
    server_data = load_mnist()
    top_model = nn.Linear(HIDDEN_DIM, 10)
    top_optimizer = optim.SGD(top_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    client = collab.clients[0]

    for step in range(NUM_STEPS):
        _, labels = get_batch(server_data, step)
        activations = client.forward(step).requires_grad_(True)

        top_optimizer.zero_grad(set_to_none=True)
        logits = top_model(activations)
        loss = criterion(logits, labels)
        loss.backward()

        client.backward(activations.grad)
        top_optimizer.step()

        with torch.no_grad():
            batch_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"  step {step + 1}/{NUM_STEPS}  loss={loss.item():.4f}  batch_acc={batch_accuracy:.4f}")

    print("Split learning finished.")
    return top_model.state_dict()


def make_recipe():
    return CollabRecipe(job_name="simple_split_learning", min_clients=1, sync_task_timeout=60)


def main():
    simple_logging()
    run = make_recipe().execute(SimEnv(num_clients=1))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
