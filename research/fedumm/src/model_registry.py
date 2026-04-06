# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Model registry for federated VLM training.

Each backend registers itself by calling :func:`register_backend`.
The unified FL client selects a backend via ``--model_backend <name>``.

Adding a new model:
  1. Create ``src/<model>_backend.py`` with a class implementing the
     same interface as :class:`BLIPBackend` / :class:`JanusProBackend`.
  2. Call ``register_backend(name, instance)`` at module import time.
  3. Import the module in ``src/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

_REGISTRY: Dict[str, Any] = {}


def register_backend(name: str, backend: Any) -> None:
    _REGISTRY[name.lower()] = backend


def get_backend(name: str) -> Any:
    key = name.lower()
    if key not in _REGISTRY:
        avail = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available: {avail}")
    return _REGISTRY[key]


def list_backends() -> List[str]:
    return sorted(_REGISTRY.keys())
