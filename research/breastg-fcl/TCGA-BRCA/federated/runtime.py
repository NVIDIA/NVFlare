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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Transport-independent client operations with isolated random-number state."""

import hashlib
import json
import random
import threading
from contextlib import contextmanager
from numbers import Integral

import numpy as np
import torch

# NVFlare executors may share a process. Serialize seeded contexts so one
# operation cannot overwrite another operation's process-global RNG state.
_RNG_LOCK = threading.RLock()
_OPERATIONS = frozenset(("encode", "train", "test"))


def operation_seed(base_seed, task, round_index, client_id, operation):
    """Derive a stable 32-bit seed independently of worker assignment or order."""
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown client operation: {operation!r}")
    identity = json.dumps(
        [int(base_seed), int(task), int(round_index), int(client_id), operation],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(identity, digest_size=4).digest(), "big")


def snapshot_rng(device=None):
    """Save Python, NumPy, CPU Torch, and optionally one CUDA device's RNG."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda_device": None,
        "cuda": None,
    }
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            state["cuda_device"] = index
            state["cuda"] = torch.cuda.get_rng_state(index).clone()
    return state


def restore_rng(state):
    """Restore a snapshot without touching unrelated CUDA devices."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda_device"] is not None:
        torch.cuda.set_rng_state(state["cuda"], state["cuda_device"])


@contextmanager
def seeded_rng(seed=None, device=None):
    """Run with an optional local seed and restore the caller's RNG on exit."""
    with _RNG_LOCK:
        state = snapshot_rng(device)
        try:
            if seed is not None:
                seed = int(seed)
                random.seed(seed)
                np.random.seed(seed % (1 << 32))
                # torch.manual_seed also seeds every CUDA device; use the CPU
                # generator directly, then seed only the client's CUDA device.
                torch.random.default_generator.manual_seed(seed % (1 << 64))
                if state["cuda_device"] is not None:
                    with torch.cuda.device(state["cuda_device"]):
                        torch.cuda.manual_seed(seed % (1 << 64))
            yield
        finally:
            restore_rng(state)


def execute_client(operation, client, task, relational_graphs, dataloader, epochs=1, seed=None):
    """Execute an E/F/G operation on an NVFlare client.

    ``encode`` returns client-noised latents and matching graph rows;
    ``train`` returns E/F/G weights
    and client optimizer/task state; ``test`` returns the client's metrics.
    Model state is intentionally updated by training, while process RNG state
    is always restored, including when a client operation raises an exception.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown client operation: {operation!r}")
    if operation == "train" and (not isinstance(epochs, Integral) or epochs < 1):
        raise ValueError("Client training requires a positive integer epoch count")
    with seeded_rng(seed, getattr(client, "device", None)):
        if operation == "encode":
            return client.generate_encodings(task, relational_graphs, dataloader, False)
        if operation == "test":
            return client.test(task, dataloader, relational_graphs)
        for epoch in range(epochs):
            client.learn(epoch, task, relational_graphs, dataloader, False)
        result = client.get_weights()
        result["training_state"] = client.get_training_state()
        return result
