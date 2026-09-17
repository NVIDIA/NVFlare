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

"""Published CIFAR-10 client training method for the FedRevive workflow."""

import ctypes
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from data import train_transform
from model import create_model, get_model_params, load_model_params
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


def _trim_process_memory():
    """Best-effort return of freed allocator pages to the host OS.

    CPython and the native allocator may keep large deserialized model buffers
    in their arenas after an RPC ends.  Reusing a physical process for many
    logical clients can therefore grow its RSS even when Python objects have
    been collected.  ``malloc_trim`` prevents that retained high-water mark
    from accumulating into host-wide memory pressure on Linux; other platforms
    safely fall back to normal allocator behavior.
    """

    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


class FedReviveClient:
    """Train prepared logical clients with one reusable physical-worker model.

    The paper simulates K=100 concurrent *logical* clients, but creating that
    many local NVFlare processes, dataset copies, thread pools, and CUDA
    contexts can exhaust a workstation.  A small pool of physical Collab sites
    instead multiplexes assignments.  Logical state lives in prepared shards
    and the server manifest, so recycling a worker does not alter participant
    selection or the delay-driven arrival sequence.
    """

    def __init__(
        self,
        data_root: str,
        prepared_data_root: str,
        local_batch_size: int = 32,
        local_iterations: int = 25,
        local_lr: float = 3e-4,
        device: str | None = "cpu",
        num_threads: int = 1,
        num_workers: int = 0,
    ):
        self.data_root = data_root
        self.prepared_data_root = prepared_data_root
        self.local_batch_size = local_batch_size
        self.local_iterations = local_iterations
        self.local_lr = local_lr
        self.device = device
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.logger = get_obj_logger(self)
        self._dataset = None
        self._model = None
        self._indices = {}

    @collab.init
    def initialize(self):
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(self.num_threads)
        self.data_root = collab.get_app_prop("data_root", self.data_root)
        self.prepared_data_root = collab.get_app_prop("prepared_data_root", self.prepared_data_root)
        self._dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=train_transform(),
        )
        self._model = create_model()
        self.logger.info(f"[{collab.site_name}] initialized FedRevive CIFAR-10 client")

    def _device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _logical_indices(self, logical_name: str) -> np.ndarray:
        # A logical client is represented by a small index vector on disk, not
        # a resident dataset/model/optimizer process.  Index vectors are cheap
        # to cache; image storage remains shared in the single CIFAR-10 dataset.
        if logical_name not in self._indices:
            path = Path(self.prepared_data_root).expanduser().resolve() / "splits" / f"{logical_name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"No prepared shard for logical client {logical_name}: {path}")
            self._indices[logical_name] = np.load(path)
        return self._indices[logical_name]

    def _loader(self, logical_name: str, train_seed: int) -> DataLoader:
        # Assignment-derived seeds make minibatch order follow the logical job
        # rather than whichever physical site happens to execute it.
        generator = torch.Generator().manual_seed(train_seed)
        subset = Subset(self._dataset, self._logical_indices(logical_name).tolist())
        return DataLoader(
            subset,
            batch_size=self.local_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
        )

    # @collab.publish exposes this method for the server's nonblocking
    # collab.get_clients(...).train(...) calls.  The dictionaries are keyed by
    # physical site because one group call may carry distinct logical clients,
    # assignment ids, and seeds to each target.
    @collab.publish
    def train(
        self,
        assignment_ids: dict[str, int],
        model_version: int,
        global_model: dict[str, torch.Tensor],
        logical_assignments: dict[str, str],
        train_seeds: dict[str, int],
    ):
        if collab.is_aborted:
            return None

        # The preceding RPC has fully unwound by now.  Reclaim its serialized
        # model buffers before accepting another logical-client assignment;
        # trimming earlier could race with response serialization.
        _trim_process_memory()

        physical_name = collab.site_name
        logical_name = logical_assignments[physical_name]
        assignment_id = int(assignment_ids[physical_name])
        train_seed = int(train_seeds[physical_name])
        torch.manual_seed(train_seed)

        device = self._device()
        started = time.perf_counter()
        model = load_model_params(self._model, global_model, target_device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.local_lr)
        criterion = nn.CrossEntropyLoss()
        loader = self._loader(logical_name, train_seed)
        iterator = iter(loader)

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for _ in range(self.local_iterations):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, labels = next(iterator)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total_samples += labels.size(0)

        # Return an independent CPU snapshot.  Keeping response tensors off GPU
        # prevents completed RPCs from pinning scarce device memory while the
        # server waits for their simulated upload events.
        updated_model = get_model_params(model, target_device="cpu")
        metrics = {
            "train_loss": total_loss / self.local_iterations,
            "train_accuracy": total_correct / total_samples,
            "num_steps": self.local_iterations,
            "num_samples": total_samples,
            "train_time": time.perf_counter() - started,
            "device": str(device),
        }
        metadata = {
            "assignment_id": assignment_id,
            "model_version": int(model_version),
            "logical_name": logical_name,
            "train_seed": train_seed,
        }
        self.logger.info(
            f"[{collab.call_info}] assignment={assignment_id} logical={logical_name} "
            f"base_version={model_version} loss={metrics['train_loss']:.4f}"
        )

        # Logical-client state is the prepared shard on disk plus its manifest
        # profile.  A fresh Adam optimizer is used for every assignment, so it
        # can be discarded here without losing client state.  DataLoader and
        # iterator deletion also closes any per-assignment loader workers.  Keep
        # only one reusable model on CPU between calls and release CUDA caches;
        # this bounds resources by physical sites instead of logical K.
        model.to("cpu")
        del optimizer, loader, iterator, inputs, labels, outputs, loss, criterion
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _trim_process_memory()
        return updated_model, metrics, metadata
