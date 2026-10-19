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

"""Epoch-boundary state for the CIFAR examples' map-style data loaders."""

import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, default_collate, get_worker_info


def _capture_worker_batch(samples):
    batch = default_collate(samples)
    numpy_rng = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:]),
        "torch": torch.get_rng_state(),
    }
    return batch, get_worker_info().id, state


def _restore_worker(worker_id, states):
    state = states.get(worker_id)
    if state is None:
        return
    random.setstate(state["python"])
    numpy_rng = state["numpy"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(state["torch"])


class CheckpointDataLoader:
    """Retain sampler and worker RNG state after a fully consumed local epoch.

    The clients checkpoint only after all local epochs finish. At that boundary
    every prefetched batch has been consumed, including each worker's last RNG
    snapshot. Job lifetime keeps its workers; fresh task workers initialize from
    the same snapshots. Native sampler/model RNG coupling and job-lifetime seed
    draws are preserved. Recreating persistent workers avoids an extra draw from
    that main-process stream.
    This is not a mid-epoch or arbitrary-dataset checkpoint implementation.
    """

    def __init__(self, dataset, *, batch_size, shuffle, num_workers, pin_memory):
        self.dataset = dataset
        self._started = False
        self._worker_seed_rng = torch.Generator()
        self._worker_states = {}
        self._num_workers = num_workers
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RandomSampler(dataset) if shuffle else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=self._worker_seed_rng,
            collate_fn=_capture_worker_batch if num_workers else None,
            worker_init_fn=partial(_restore_worker, states=self._worker_states) if num_workers else None,
        )

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        # Native loaders draw one base seed for each new iterator. Persistent
        # job workers reuse theirs across epochs. A recreated task pool uses an
        # isolated copy for its physical seed while restored worker RNG wins.
        self._worker_seed_rng.set_state(torch.get_rng_state())
        if not self._started or not self._num_workers:
            torch.empty((), dtype=torch.int64).random_()
        self._started = True
        for batch in self._loader:
            if self._num_workers:
                batch, worker_id, state = batch
                self._worker_states[worker_id] = state
            yield batch

    def state_dict(self):
        return {
            "num_workers": self._num_workers,
            "started": self._started,
            "workers": self._worker_states,
        }

    def load_state_dict(self, state):
        if state["num_workers"] != self._num_workers:
            raise RuntimeError("Training checkpoint DataLoader worker count changed")
        self._started = state["started"]
        self._worker_states.clear()
        self._worker_states.update(state["workers"])
