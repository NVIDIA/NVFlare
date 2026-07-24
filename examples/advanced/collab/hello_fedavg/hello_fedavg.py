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

Per-site configuration: ``recipe.set_per_site_config({site: {name: value}})``
resolves values for each site before execution, so a client receives only its
own values and reads them with ``collab.get_app_prop(name)``. Here each site
trains for a different number of local epochs.

Run:
    python -m collab.hello_fedavg.hello_fedavg
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class Trainer:
    @collab.publish
    def train(self, weights=None):
        # The recipe resolves this site's value before the client starts.
        local_epochs = collab.get_app_prop("local_epochs", 5)

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
    valid = dict(client_results)
    for client_id, error in client_results.failures.items():
        print(f"  Warning: {client_id} failed: {error}")
    if not valid:
        raise RuntimeError(f"all {len(client_results.failures)} client calls failed")

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
    recipe.set_per_site_config(
        {f"site-{site_num}": {"local_epochs": 2 if site_num == 1 else 5} for site_num in range(1, args.num_clients + 1)}
    )
    return recipe


def main():
    parser = argparse.ArgumentParser(description="Hello-world FedAvg with the Collab API")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    args = parser.parse_args()
    simple_logging()
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
