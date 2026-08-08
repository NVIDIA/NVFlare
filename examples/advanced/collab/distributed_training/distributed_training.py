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

"""Federated averaging with one persistent PyTorch DDP rank group per client.

Run from ``examples/advanced`` on a CUDA system with at least two GPUs:

    python -m collab.distributed_training.distributed_training

``CollabRecipe`` treats each site's command as a launcher prefix and appends
the Collab distributed worker module. The resulting client lifecycle is SPMD:
``initialize``, every ``train`` call, and ``finalize`` run on every rank, while
only global rank zero's return value crosses the federated boundary.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

NUM_FEATURES = 8
NUM_SAMPLES = 256
BATCH_SIZE = 32


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, 1)

    def forward(self, inputs):
        return self.linear(inputs)


class DDPTrainer:
    def __init__(self):
        self.device = None
        self.dataset = None
        self.local_epochs = 1

    @collab.init
    def initialize(self):
        """Initialize NCCL and rank-local state in every torchrun process."""
        if not torch.cuda.is_available():
            raise RuntimeError("this example requires CUDA")
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("LOCAL_RANK is required; launch the client with torchrun")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")

        self.local_epochs = int(collab.get_app_prop("local_epochs", 2))
        data_seed = int(collab.get_app_prop("data_seed", 1000))
        generator = torch.Generator().manual_seed(data_seed)
        inputs = torch.randn(NUM_SAMPLES, NUM_FEATURES, generator=generator)
        target_weights = torch.arange(1, NUM_FEATURES + 1, dtype=torch.float32) / NUM_FEATURES
        noise = 0.05 * torch.randn(NUM_SAMPLES, generator=generator)
        targets = inputs @ target_weights + 0.25 + noise
        self.dataset = TensorDataset(inputs, targets.unsqueeze(1))

        if dist.get_rank() == 0:
            print(
                f"  [{collab.site_name}] initialized {dist.get_world_size()} DDP ranks "
                f"with {self.local_epochs} local epochs"
            )

    @collab.publish
    def train(self, weights, round_number):
        """Run one federated round on every rank and return rank zero's model."""
        model = RegressionModel().to(self.device)
        model.load_state_dict(weights)
        model = DDP(model, device_ids=[self.device.index])

        sampler = DistributedSampler(
            self.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, sampler=sampler)
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        total_loss = torch.zeros(1, device=self.device)
        total_samples = torch.zeros(1, device=self.device)
        for epoch in range(self.local_epochs):
            sampler.set_epoch(round_number * self.local_epochs + epoch)
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * inputs.size(0)
                total_samples += inputs.size(0)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        mean_loss = (total_loss / total_samples).item()

        # The Collab runtime invokes this method on every rank and forwards only
        # global rank zero's result. Returning None elsewhere also makes that
        # application contract explicit.
        if dist.get_rank() != 0:
            return None

        result = {name: tensor.detach().cpu().clone() for name, tensor in model.module.state_dict().items()}
        print(f"  [{collab.site_name}] round={round_number + 1} loss={mean_loss:.4f}")
        return result, mean_loss

    @collab.final
    def finalize(self):
        """Run normal DDP teardown on every rank."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


class FedAvg:
    def __init__(self, num_rounds):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        """Call every distributed client and average their rank-zero models."""
        torch.manual_seed(0)
        global_weights = RegressionModel().state_dict()

        for round_number in range(self.num_rounds):
            print(f"=== Federated round {round_number + 1} ===")
            client_results = collab.clients.train(global_weights, round_number)
            valid_results = dict(client_results)
            for client_name, error in client_results.failures.items():
                print(f"  Warning: {client_name} failed: {error}")
            if not valid_results:
                raise RuntimeError("all distributed clients failed")

            client_weights = [result[0] for result in valid_results.values()]
            global_weights = {
                name: torch.stack([weights[name] for weights in client_weights]).mean(dim=0)
                for name in client_weights[0]
            }
            mean_loss = sum(result[1] for result in valid_results.values()) / len(valid_results)
            print(f"  Global average loss: {mean_loss:.4f}")

        return global_weights


def make_recipe(args):
    recipe = CollabRecipe(
        job_name="collab_distributed_training",
        server=FedAvg(num_rounds=args.num_rounds),
        client=DDPTrainer(),
        min_clients=args.num_clients,
        sync_task_timeout=120,
        launch_external_process=True,
        distributed_startup_timeout=120,
    )
    recipe.set_per_site_config(
        {
            f"site-{site_number}": {
                "command": (
                    "python3 -m torch.distributed.run --nnodes=1 "
                    f"--nproc_per_node={args.nproc_per_client} "
                    f"--master_port={args.master_port + site_number - 1}"
                ),
                "data_seed": 1000 + site_number,
                "local_epochs": args.local_epochs,
            }
            for site_number in range(1, args.num_clients + 1)
        }
    )
    return recipe


def define_parser():
    parser = argparse.ArgumentParser(description="Collab federated learning with per-client PyTorch DDP")
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--nproc-per-client", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29500, help="First site's torchrun rendezvous port")
    parser.add_argument("--export-config", action="store_true")
    parser.add_argument("--log-config", default="concise")
    return parser


def main():
    args = define_parser().parse_args()
    simple_logging()
    recipe = make_recipe(args)

    if args.export_config:
        job_dir = "/tmp/nvflare/jobs/collab_distributed_training"
        recipe.export(job_dir)
        print(f"Job config exported to {job_dir}")
        return

    env = SimEnv(clients=recipe.configured_sites(), num_threads=args.num_clients, log_config=args.log_config)
    run = recipe.execute(env)
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
