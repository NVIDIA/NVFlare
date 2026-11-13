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

"""Research PyTorch RNG helpers; importing SeedPolicy alone needs no torch."""

import random
import threading
from contextlib import contextmanager

import numpy as np
import torch

from .seed_policy import _check_seed

# These libraries expose process-global RNGs. Serialize cooperating scopes;
# unrelated threads must not use those global RNGs concurrently with a scope.
_rng_lock = threading.RLock()


@contextmanager
def torch_seed_scope(seed: int, *, cuda_devices=()):
    """Temporarily seed Python, NumPy, CPU torch and explicitly selected GPUs.

    Restore all selected RNG states even after an exception. The CPU-only
    default neither initializes nor seeds CUDA. Pass every CUDA device used
    inside the scope; other devices are untouched. Existing Generator objects
    remain independent and must be seeded explicitly. Algorithm/backend flags
    are not changed, and this does not provide distributed checkpoint recovery.
    """
    _check_seed(seed)
    devices = list(cuda_devices)
    if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in devices):
        raise ValueError("cuda_devices must contain nonnegative device indices")
    devices = list(dict.fromkeys(devices))
    with _rng_lock:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed % 2**32)
                # torch.manual_seed also touches every CUDA device; seed CPU
                # directly so CPU jobs and other GPU streams remain isolated.
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def seed_worker(worker_id: int):
    """DataLoader worker_init_fn: extend its torch seed to Python and NumPy.

    A loader's explicit Generator determines torch.initial_seed() for each
    worker. Recreate workers when an assignment seed changes; persistent
    workers do not run this initializer again for the next assignment.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
