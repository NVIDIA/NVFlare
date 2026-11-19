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

"""Stateless, assignment-seeded CIFAR-10 trainer for physical-worker replay."""

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pt_async_cifar10_reproducible.model import ModerateCNN, get_model_params, load_model_params
from pt_async_cifar10_reproducible.pt_seed import seed_worker, torch_seed_scope
from pt_async_cifar10_reproducible.seed_policy import SeedPolicy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        ),
    ]
)


class Cifar10Trainer:
    """Reset optimizer and data-loader state for each logical assignment."""

    def __init__(
        self,
        data_root: str,
        local_batch_size: int = 64,
        local_lr: float = 0.05,
        device: str | None = None,
        num_threads: int = 1,
        train_idx_root: str | None = None,
        aggregation_epochs: int = 4,
        total_client_rounds: int = 50,
        num_workers: int = 0,
        num_logical_clients: int = 8,
        run_seed: int = 10,
    ):
        self.data_root = data_root
        self.local_batch_size = local_batch_size
        self.local_lr = local_lr
        self.device = device
        self.num_threads = num_threads
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.total_client_rounds = total_client_rounds
        self.num_workers = num_workers
        self.num_logical_clients = num_logical_clients
        self.run_seed = run_seed
        self.logger = get_obj_logger(self)
        self._dataset = None
        self._local_model = None

    @collab.init
    def initialize(self):
        torch.set_num_threads(self.num_threads)
        self.data_root = collab.get_app_prop("data_root", self.data_root)
        self._dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=_TRAIN_TRANSFORM,
        )
        self._local_model = ModerateCNN()
        self.logger.info(f"[{collab.site_name}] loaded assignment-seeded CIFAR-10 training settings")

    def _device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _build_dataloader(self, logical_name: str, seed: int) -> DataLoader:
        indices_file = Path(self.train_idx_root).expanduser().resolve() / f"{logical_name}.npy"
        if not indices_file.is_file():
            raise FileNotFoundError(
                f"No prepared split for logical client '{logical_name}' at '{indices_file}'. "
                "Re-run job.py to regenerate the shared CIFAR-10 split."
            )
        indices = np.load(indices_file)
        return DataLoader(
            Subset(self._dataset, indices.tolist()),
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(SeedPolicy(seed).seed("data")),
            worker_init_fn=seed_worker,
        )

    def _learning_rate(self, assignment_id: int) -> float:
        # This assignment-index schedule is independent of which physical site runs it.
        local_round = min(assignment_id // self.num_logical_clients, self.total_client_rounds)
        eta_min = self.local_lr * 0.01
        return eta_min + 0.5 * (self.local_lr - eta_min) * (
            1 + math.cos(math.pi * local_round / self.total_client_rounds)
        )

    def _prepare_model(self, global_model: dict[str, torch.Tensor], device: torch.device):
        load_model_params(self._local_model, global_model, target_device=device)
        return self._local_model

    # @collab.publish exposes this method to the server's nonblocking
    # collab.get_clients(...).train(...) calls.
    @collab.publish
    def train(
        self,
        assignment_ids: dict[str, int],
        model_version: int,
        global_model: dict[str, torch.Tensor],
        logical_assignments: dict[str, str],
        train_seeds: dict[str, int] | None = None,
    ):
        started = time.time()
        if collab.is_aborted:
            self.logger.info(f"[{collab.site_name}] training aborted")
            return None

        physical_name = collab.site_name
        logical_name = logical_assignments.get(physical_name, physical_name)
        assignment_id = assignment_ids[physical_name]
        seed = (
            train_seeds[physical_name]
            if train_seeds is not None
            else SeedPolicy(self.run_seed).seed("train", assignment_id=assignment_id, logical_name=logical_name)
        )
        self.logger.info(
            f"[{collab.call_info}] starting assignment {assignment_id} from model version {model_version} "
            f"(physical={physical_name}, logical={logical_name})"
        )

        device = self._device()
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        )
        with torch_seed_scope(seed, cuda_devices=cuda_devices):
            data_started = time.time()
            dataloader = self._build_dataloader(logical_name, seed)
            data_time = time.time() - data_started

            model_started = time.time()
            model = self._prepare_model(global_model, device)
            optimizer = torch.optim.SGD(model.parameters(), lr=self._learning_rate(assignment_id), momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            model_time = time.time() - model_started

            train_started = time.time()
            total_loss = 0.0
            num_steps = self.aggregation_epochs * len(dataloader)
            if num_steps == 0:
                raise ValueError(f"Prepared shard for '{logical_name}' contains no training batches")
            for _epoch in range(self.aggregation_epochs):
                model.train()
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            train_time = time.time() - train_started
            updated_state = get_model_params(model, target_device="cpu")

        metrics = {
            "train_loss": total_loss / num_steps,
            "num_samples": len(dataloader.dataset),
            "num_steps": num_steps,
        }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self.logger.info(
            f"[{collab.call_info}] completed assignment {assignment_id} (logical={logical_name}); "
            f"loss={metrics['train_loss']:.4f}; data={data_time:.3f}s, model={model_time:.3f}s, "
            f"training={train_time:.3f}s, total={time.time() - started:.3f}s"
        )
        metadata = {
            "assignment_id": assignment_id,
            "model_version": model_version,
            "logical_name": logical_name,
            "train_seed": seed,
        }
        return updated_state, metrics, metadata
