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

"""Stateless, named random streams reusable without a Collab runtime.

Seed derivation never consumes a generator or changes process RNG state.
Only SeededTrainArgs adapts the policy to the asynchronous train protocol.
"""

import hashlib
import json
import random


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return seed


class SeedPolicy:
    """Derive stable 63-bit seeds from a root, named stream and optional job.

    The v1 derivation hashes an explicitly encoded identity, rather than using
    Python hash() or process/worker identity. Adding a new stream or consuming
    more randomness in KD does not change training seeds. Explicit generators
    are caller-owned; calling a generator factory again restarts that stream.
    This selects randomness; it does not enforce deterministic model kernels.
    """

    def __init__(self, run_seed: int):
        self.run_seed = _check_seed(run_seed)

    def seed(self, stream: str, *, assignment_id: int | None = None, logical_name: str | None = None) -> int:
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be a nonempty string")
        if assignment_id is not None and (
            isinstance(assignment_id, bool) or not isinstance(assignment_id, int) or assignment_id < 0
        ):
            raise ValueError("assignment_id must be a nonnegative integer or None")
        if logical_name is not None and (not isinstance(logical_name, str) or not logical_name):
            raise ValueError("logical_name must be a nonempty string or None")
        identity = ["nvflare.seed.v1", self.run_seed, stream, assignment_id, logical_name]
        encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (2**63 - 1)

    def python_rng(self, stream, **scope):
        """Return a fresh independent random.Random; never seed the global RNG."""
        return random.Random(self.seed(stream, **scope))

    def numpy_rng(self, stream, **scope):
        """Return a fresh independent NumPy Generator; NumPy is imported lazily."""
        import numpy as np

        return np.random.default_rng(self.seed(stream, **scope))


class SeededTrainArgs:
    """Use with AsyncRecipe(train_kwargs=...) to seed logical assignments.

    An explicit assignment.metadata['train_seed'] takes precedence (trace
    replay). Otherwise derive it from logical identity and assignment ID. Stamp
    the selected seed into server-owned metadata before dispatch so on_event
    records it. No physical-worker name participates in seed derivation.
    The client must declare train_seeds and apply its own seed/RNG context.
    """

    def __init__(self, seed_policy: SeedPolicy, stream: str = "train"):
        if not isinstance(seed_policy, SeedPolicy):
            raise TypeError("seed_policy must be a SeedPolicy")
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be a nonempty string")
        self.seed_policy = seed_policy
        self.stream = stream

    def __call__(self, call):
        assignment = call.assignment
        if "train_seed" in assignment.metadata:
            seed = _check_seed(assignment.metadata["train_seed"])
        else:
            seed = self.seed_policy.seed(
                self.stream, assignment_id=assignment.assignment_id, logical_name=assignment.logical_name
            )
            assignment.metadata["train_seed"] = seed
        return {"train_seeds": {call.client_name: seed}}
