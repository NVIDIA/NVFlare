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

"""Hello-world: federated averaging with the Collab API.

The server drives rounds from a single ``@collab.main`` method; each client
exposes ``train`` via ``@collab.publish``; ``collab.clients.train(...)`` fans
the call out to every client in parallel and returns their results.

Server and client do not have to be classes: ``@collab.main`` and
``@collab.publish`` also work on plain module-level functions, in which case
``CollabRecipe`` picks up the current module automatically when ``server``/
``client`` are omitted.

Per-site configuration: ``recipe.set_client_prop(name, {site: value})`` ships
a per-site value that each site reads at run time with
``collab.get_app_prop(name)`` (sites not listed fall back to a default). Here
each site trains for a different number of local epochs.

Every recipe example also takes a ``--runtime`` option (in_process |
multi_process | prod | export); the recipe itself is identical across them.

Run:
    python -m collab.hello_fedavg.hello_fedavg
    python -m collab.hello_fedavg.hello_fedavg --runtime multi_process
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collab.common.runner import make_parser, run_recipe
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import CollabRecipe, collab


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class Trainer:
    @collab.publish
    def train(self, weights=None):
        # Per-site config: local epochs are configured per site on the recipe
        # and read here at run time (default 5 if this site isn't configured).
        local_epochs = collab.get_app_prop("local_epochs", {}).get(collab.site_name, 5)

        inputs = torch.randn(100, 10)
        labels = torch.randn(100, 1)
        dataloader = DataLoader(TensorDataset(inputs, labels), batch_size=10)

        model = SimpleModel()
        if weights is not None:
            model.load_state_dict(weights)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for _epoch in range(local_epochs):
            for batch_inputs, batch_labels in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(batch_inputs), batch_labels)
                loss.backward()
                optimizer.step()

        print(f"  [{collab.site_name}] epochs={local_epochs} Loss: {loss.item():.4f}")
        return model.state_dict(), loss.item()


def weighted_avg(client_results):
    valid = {}
    for client_id, result in client_results:
        if isinstance(result, Exception):
            print(f"  Warning: {client_id} failed: {result}")
            continue
        valid[client_id] = result
    if not valid:
        raise RuntimeError(f"all {len(client_results)} client calls failed")

    all_weights = [result[0] for result in valid.values()]
    avg_weights = {k: torch.stack([w[k] for w in all_weights]).mean(dim=0) for k in all_weights[0]}
    avg_loss = sum(result[1] for result in valid.values()) / len(valid)
    return avg_weights, avg_loss


class FedAvg:
    def __init__(self, num_rounds=3):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        global_weights = None
        for round_num in range(self.num_rounds):
            print(f"=== Round {round_num + 1} ===")
            client_results = collab.clients.train(global_weights)
            global_weights, global_loss = weighted_avg(client_results)
            print(f"  Global average loss: {global_loss:.4f}")
        return global_weights


def make_recipe(args):
    recipe = CollabRecipe(
        job_name="hello_fedavg",
        server=FedAvg(num_rounds=args.num_rounds),
        client=Trainer(),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )
    # Per-site configuration: site-1 trains fewer local epochs than site-2.
    recipe.set_client_prop("local_epochs", {"site-1": 2, "site-2": 5})
    return recipe


def main():
    parser = make_parser("Hello-world FedAvg with the Collab API")
    parser.add_argument("--num-rounds", type=int, default=3)
    args = parser.parse_args()
    run_recipe(make_recipe(args), args)


if __name__ == "__main__":
    main()
