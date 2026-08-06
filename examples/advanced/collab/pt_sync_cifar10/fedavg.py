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

"""Synchronous CIFAR-10 FedAvg using direct Collab function calls."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from data import DEFAULT_DATA_ROOT, make_test_loader, make_train_loader, validate_prepared_data
from model import Cifar10CNN, get_model_state

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_sync_cifar10_fedavg"
EXAMPLE_DIR = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str | None) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))


def collect_client_updates(client_results, min_clients: int) -> dict:
    updates = dict(client_results)
    for site_name, error in getattr(client_results, "failures", {}).items():
        print(f"Warning: {site_name} failed: {error}")
    if len(updates) < min_clients:
        raise RuntimeError(f"Received {len(updates)} client updates; need at least {min_clients}")
    return updates


def weighted_average(client_results, min_clients: int) -> tuple[dict[str, torch.Tensor], float]:
    updates = collect_client_updates(client_results, min_clients)

    first = next(iter(updates.values()))
    expected_keys = set(first["weights"])
    total_weight = sum(update["num_steps"] for update in updates.values())
    if total_weight <= 0:
        raise RuntimeError("Client updates contain no training steps")

    for site_name, update in updates.items():
        if update["num_steps"] <= 0:
            raise ValueError(f"{site_name} returned an invalid step count")
        if set(update["weights"]) != expected_keys:
            raise ValueError(f"{site_name} returned a different model state")

    averaged = {}
    for name in sorted(expected_keys):
        reference = first["weights"][name]
        if not reference.is_floating_point():
            averaged[name] = reference.detach().cpu().clone()
            continue

        accumulator = torch.zeros_like(reference, device="cpu", dtype=torch.float32)
        for update in updates.values():
            value = update["weights"][name]
            if value.shape != reference.shape:
                raise ValueError(f"Client returned an incompatible shape for {name}")
            accumulator.add_(value.detach().to(device="cpu", dtype=torch.float32), alpha=update["num_steps"])
        averaged[name] = (accumulator / total_weight).to(reference.dtype)

    average_loss = sum(update["train_loss"] * update["num_steps"] for update in updates.values()) / total_weight
    return averaged, average_loss


class FedAvgClient:
    def __init__(
        self,
        data_root: str,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        momentum: float,
        num_rounds: int,
        num_workers: int,
        device: str | None = None,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_rounds = num_rounds
        self.num_workers = num_workers
        self.device_name = device
        self.seed = seed
        self.device = None
        self.model = None
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None

    def _prepare_local_training(self) -> None:
        """Hook for algorithms that need state from the received global model."""

    def _compute_loss(self, inputs, labels, criterion) -> torch.Tensor:
        return criterion(self.model(inputs), labels)

    def _after_optimizer_step(self, optimizer) -> None:
        """Hook for algorithms that adjust the model after each optimizer step."""

    @collab.init
    def initialize(self):
        seed_everything(self.seed)
        self.device = resolve_device(self.device_name)
        self.model = Cifar10CNN().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_rounds * self.local_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        self.train_loader = make_train_loader(
            self.data_root,
            collab.site_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        print(f"[{collab.site_name}] {len(self.train_loader.dataset)} training examples on {self.device}")

    @collab.publish
    def train(self, round_number: int, global_weights: dict[str, torch.Tensor]) -> dict:
        if collab.is_aborted:
            raise RuntimeError("Training aborted")

        self.model.load_state_dict(global_weights, strict=True)
        train_loss, local_steps = self._local_train()
        print(f"[{collab.site_name}] round={round_number} train_loss={train_loss:.4f}")
        return {
            "weights": get_model_state(self.model),
            "train_loss": train_loss,
            "num_steps": local_steps,
        }

    def _local_train(self) -> tuple[float, int]:
        if len(self.train_loader) == 0:
            raise ValueError("Training data loader contains no examples; check the prepared client split")
        criterion = nn.CrossEntropyLoss()
        self._prepare_local_training()
        self.model.train()

        loss_sum = 0.0
        examples_seen = 0
        local_steps = 0
        for _epoch in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                if collab.is_aborted:
                    raise RuntimeError("Training aborted")
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                loss = self._compute_loss(inputs, labels, criterion)
                loss.backward()
                self.optimizer.step()
                self._after_optimizer_step(self.optimizer)
                batch_size = labels.size(0)
                loss_sum += loss.item() * batch_size
                examples_seen += batch_size
                local_steps += 1
            self.scheduler.step()

        if examples_seen == 0:
            raise ValueError("Training data loader contains no examples; check the prepared client split")
        return loss_sum / examples_seen, local_steps


class FedAvgServer:
    def __init__(
        self,
        data_root: str,
        output_root: str,
        num_rounds: int,
        min_clients: int,
        eval_batch_size: int,
        num_workers: int,
        device: str | None = None,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.output_root = output_root
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.device_name = device
        self.seed = seed

    def evaluate(self, model: nn.Module, test_loader) -> float:
        model.eval()
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                correct += model(inputs).argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)
        return correct / total

    def save_outputs(self, global_weights, history) -> None:
        output_root = Path(self.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        model_path = output_root / "model_final.pt"
        torch.save(global_weights, model_path)
        with (output_root / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
            stream.write("\n")
        print(f"Saved final model to {model_path}")

    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        seed_everything(self.seed)
        device = resolve_device(self.device_name)
        model = Cifar10CNN().to(device)
        test_loader = make_test_loader(self.data_root, self.eval_batch_size, self.num_workers)
        global_weights = get_model_state(model)
        history = [{"round": 0, "test_accuracy": self.evaluate(model, test_loader)}]
        print(f"Initial test_accuracy={history[0]['test_accuracy']:.4f}")

        for round_number in range(1, self.num_rounds + 1):
            print(f"=== FedAvg round {round_number}/{self.num_rounds} ===")
            client_results = collab.clients.train(round_number, global_weights)
            global_weights, train_loss = weighted_average(client_results, self.min_clients)
            model.load_state_dict(global_weights, strict=True)
            accuracy = self.evaluate(model, test_loader)
            history.append({"round": round_number, "test_accuracy": accuracy, "train_loss": train_loss})
            print(f"Aggregated train_loss={train_loss:.4f}, test_accuracy={accuracy:.4f}")

        self.save_outputs(global_weights, history)
        return global_weights


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default="/tmp/nvflare/collab/pt_sync_cifar10/fedavg")
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--call-timeout", type=float, default=3600.0)
    return parser


def make_recipe(
    args,
    server_class=FedAvgServer,
    client_class=FedAvgClient,
    client_options: dict | None = None,
    job_name: str = JOB_NAME,
    extra_files=(),
) -> CollabRecipe:
    device = None if args.device == "auto" else args.device
    recipe = CollabRecipe(
        job_name=job_name,
        server=server_class(
            data_root=args.data_root,
            output_root=args.output_root,
            num_rounds=args.num_rounds,
            min_clients=args.num_clients,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        ),
        client=client_class(
            data_root=args.data_root,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            num_workers=args.num_workers,
            device=device,
            num_rounds=args.num_rounds,
            seed=args.seed,
            **(client_options or {}),
        ),
        min_clients=args.num_clients,
        sync_task_timeout=args.call_timeout,
    )
    for path in (EXAMPLE_DIR / "data.py", EXAMPLE_DIR / "model.py", *extra_files):
        recipe.add_server_file(str(path))
        recipe.add_client_file(str(path))
    return recipe


def validate_args(args) -> None:
    if args.num_clients < 2 or args.num_rounds < 1 or args.local_epochs < 1:
        raise ValueError("--num-clients must be at least 2; --num-rounds and --local-epochs must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch sizes must be positive and --num-workers must be non-negative")
    if args.learning_rate <= 0 or not 0 <= args.momentum < 1 or args.call_timeout <= 0:
        raise ValueError("Invalid learning rate, momentum, or call timeout")


def run_example(args, recipe_factory=make_recipe, algorithm_name="FEDAVG", extra_summary=()):
    validate_args(args)
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    manifest = validate_prepared_data(args.data_root, args.num_clients)

    simple_logging(logging.INFO)
    print("=" * 80)
    print(f"CIFAR-10 COLLAB {algorithm_name}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Dirichlet alpha: {manifest['dirichlet_alpha']}")
    for label, value in extra_summary:
        print(f"  {label}: {value}")
    print(f"  Data root: {args.data_root}")
    print("=" * 80)

    run = recipe_factory(args).execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


def main():
    run_example(define_parser().parse_args())


if __name__ == "__main__":
    main()
