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

"""CIFAR-10 client trainer exposed through the current Collab API."""

import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from data import load_manifest, split_path
from model import ModerateCNN, get_model_params, load_model_params, reset_model_state, resnet18_local
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)
_FEDAVG_TRAIN_TRANSFORM = transforms.Compose(
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
    """Train logical client shards that were created by ``prepare_data.py``."""

    def __init__(
        self,
        data_root: str,
        local_batch_size: int = 32,
        local_iters: int = 25,
        local_lr: float = 0.0003,
        device: str | None = None,
        num_threads: int = 1,
        profile: str = "native",
        train_idx_root: str | None = None,
        aggregation_epochs: int = 4,
        total_client_rounds: int = 50,
        num_workers: int = 0,
    ):
        self.data_root = data_root
        self.local_batch_size = local_batch_size
        self.local_iters = local_iters
        self.local_lr = local_lr
        self.device_name = device
        self.num_threads = num_threads
        self.profile = profile
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.total_client_rounds = total_client_rounds
        self.num_workers = num_workers
        self.logger = get_obj_logger(self)
        self._dataset = None
        self._manifest = None
        self._local_model = None
        self._optimizer = None
        self._scheduler = None
        self._dataloaders = {}

    @collab.init
    def initialize(self):
        torch.set_num_threads(self.num_threads)
        self.data_root = collab.get_app_prop("data_root", self.data_root)
        if self.profile == "native":
            self._manifest = load_manifest(self.data_root)
        self._dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=_FEDAVG_TRAIN_TRANSFORM if self.profile == "fedavg" else _TRAIN_TRANSFORM,
        )
        if self.profile == "fedavg":
            self._local_model = ModerateCNN()
            self._optimizer = torch.optim.SGD(self._local_model.parameters(), lr=self.local_lr, momentum=0.9)
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self.total_client_rounds * self.aggregation_epochs,
                eta_min=self.local_lr * 0.01,
            )
            self.logger.info(f"[{collab.site_name}] loaded existing FedAvg CIFAR-10 profile")
        else:
            self.logger.info(
                f"[{collab.site_name}] loaded prepared CIFAR-10 metadata for "
                f"{self._manifest['num_clients']} logical clients"
            )

    def _device(self) -> torch.device:
        return torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _build_dataloader(self, logical_name: str, assignment_id: int) -> DataLoader:
        if self.profile == "fedavg" and logical_name in self._dataloaders:
            return self._dataloaders[logical_name]
        if self.profile == "fedavg":
            indices_file = Path(self.train_idx_root).expanduser().resolve() / f"{logical_name}.npy"
        else:
            indices_file = split_path(self.data_root, logical_name)
        if not indices_file.is_file():
            raise FileNotFoundError(
                f"No prepared split for logical client '{logical_name}' at '{indices_file}'. "
                "Re-run prepare_data.py with enough logical clients."
            )
        indices = np.load(indices_file)
        loader_args = {}
        if self.profile == "native":
            logical_index = int(logical_name.rsplit("-", 1)[-1])
            loader_args["generator"] = torch.Generator().manual_seed(
                self._manifest["split_seed"] + logical_index * 100_000 + assignment_id
            )
        dataloader = DataLoader(
            Subset(self._dataset, indices.tolist()),
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            **loader_args,
        )
        if self.profile == "fedavg":
            self._dataloaders[logical_name] = dataloader
        return dataloader

    def _prepare_model(self, global_model: dict[str, torch.Tensor], device: torch.device):
        if self._local_model is None:
            self._local_model = resnet18_local()
        if self.profile == "native":
            reset_model_state(self._local_model, reset_norm_stats=True)
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
    ):
        started = time.time()
        if collab.is_aborted:
            self.logger.info(f"[{collab.site_name}] training aborted")
            return None

        physical_name = collab.site_name
        logical_name = logical_assignments.get(physical_name, physical_name)
        assignment_id = assignment_ids[physical_name]
        self.logger.info(
            f"[{collab.call_info}] starting assignment {assignment_id} from model version {model_version} "
            f"(physical={physical_name}, logical={logical_name})"
        )

        data_started = time.time()
        dataloader = self._build_dataloader(logical_name, assignment_id)
        data_time = time.time() - data_started

        model_started = time.time()
        device = self._device()
        model = self._prepare_model(global_model, device)
        optimizer = (
            self._optimizer if self.profile == "fedavg" else torch.optim.Adam(model.parameters(), lr=self.local_lr)
        )
        criterion = nn.CrossEntropyLoss()
        model_time = time.time() - model_started

        train_started = time.time()
        model.train()
        total_loss = 0.0
        if self.profile == "fedavg":
            num_steps = self.aggregation_epochs * len(dataloader)
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
                self._scheduler.step()
        else:
            num_steps = self.local_iters
            data_iter = iter(dataloader)
            for _step in range(self.local_iters):
                try:
                    inputs, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, labels = next(data_iter)

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        train_time = time.time() - train_started

        metrics = {
            "train_loss": total_loss / num_steps,
            "num_samples": len(dataloader.dataset),
            "num_steps": num_steps,
        }
        updated_state = get_model_params(model, target_device="cpu")
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
        }
        return updated_state, metrics, metadata
