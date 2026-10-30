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

"""The `Backend` protocol — the contract every VLA adapter must satisfy.

Twelve methods total (counting properties). Implementing them all is what
makes a new VLA framework usable from decent-vla. Implementations live in
`decent_vla.adapters.<name>`.

Use `typing.Protocol` (structural), not ABC. Adapters do NOT need to
subclass; satisfying the methods is enough.

Optional methods — NOT part of the 12-method Protocol; runners discover
them via `getattr(backend, "<name>", None)` and fall back gracefully:

* `list_episodes(data_cfg: DataConfig) -> list[EpisodeMeta]`
  Episode metadata (episode_index, task_index, task_name) for episodic
  datasets, enabling the generic episode-level partitioner in
  `core/partitioner.py::partition_episodes`. Backends without
  episode structure simply omit it. MUST be callable from the CLASS
  (staticmethod or classmethod) with `data_cfg` alone: partitioning runs
  before any backend instance exists, and runners look it up via
  `getattr(backend_cls, "list_episodes", None)`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from decent_vla.core.types import Batch, CheckpointSpec, LossDict, ParamScope

if TYPE_CHECKING:
    from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig
    from decent_vla.core.policy import Policy


@runtime_checkable
class Backend(Protocol):
    """A VLA training+inference backend. One instance = one model on one device."""

    # ---------- lifecycle ----------
    @classmethod
    def from_config(cls, cfg: BackendConfig, device: torch.device) -> Backend:
        pass

    # ---------- training loop primitives (step-level) ----------
    def build_dataloader(self, data_cfg: DataConfig, partition_indices: list[int] | None) -> Iterator[Batch]:
        pass

    def compute_loss(self, batch: Batch) -> LossDict:
        pass

    def build_optimizer(self, optim_cfg: OptimConfig) -> tuple[torch.optim.Optimizer, object | None]:
        pass

    # ---------- parameter exchange (the FL boundary) ----------
    def get_state(self, scope: ParamScope = "shared") -> dict[str, torch.Tensor]:
        pass

    def set_state(self, state: dict[str, torch.Tensor], scope: ParamScope = "shared") -> None:
        pass

    # ---------- checkpointing ----------
    def save_checkpoint(self, path: str, spec: CheckpointSpec) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass

    # ---------- inference (eval path) ----------
    def build_policy(self) -> Policy:
        pass

    # ---------- metadata ----------
    @property
    def model(self) -> torch.nn.Module:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def supports_personal_params(self) -> bool:
        pass
