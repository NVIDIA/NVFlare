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

"""Tiny deterministic backend for FLARE tests and smoke runs.

A two-layer MLP maps a 16-d observation to a 7-d action with MSE loss
against random targets, allowing local FLARE training without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from src.configs.schemas import BackendConfig, DataConfig, OptimConfig
from src.support.optim import build_optimizer as make_optimizer
from src.support.parameters import load_model_checkpoint, set_model_state, trainable_state
from src.support.partitioning import EpisodeMeta


class _MLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 32, out_dim: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _synthetic_iterator(
    batch_size: int,
    in_dim: int,
    out_dim: int,
    *,
    seed: int,
    n_samples: int,
    indices: list[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yields cycling (observation, action) batches. Deterministic per (seed, indices)."""
    g = torch.Generator()
    g.manual_seed(seed)
    obs_pool = torch.randn(n_samples, in_dim, generator=g)
    act_pool = torch.randn(n_samples, out_dim, generator=g)
    if indices is not None:
        obs_pool = obs_pool[indices]
        act_pool = act_pool[indices]

    def gen() -> Iterator[dict[str, Any]]:
        i = 0
        n = obs_pool.shape[0]
        if n == 0:
            return
        while True:
            j = (i + batch_size) % n
            if j > i:
                ob = obs_pool[i:j]
                ac = act_pool[i:j]
            else:
                ob = torch.cat([obs_pool[i:], obs_pool[:j]], dim=0)
                ac = torch.cat([act_pool[i:], act_pool[:j]], dim=0)
            yield {"observation": ob, "action": ac}
            i = j

    return gen()


class MockBackend:
    """Tiny synthetic backend for tests and smoke runs."""

    IN_DIM = 16
    HIDDEN = 32
    OUT_DIM = 7
    SYNTHETIC_DATASET_SIZE = 256

    def __init__(self, cfg: BackendConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        torch.manual_seed(cfg.extra.get("model_seed", 0) if cfg.extra else 0)
        self._model = _MLP(self.IN_DIM, self.HIDDEN, self.OUT_DIM).to(device)
        from src.support.parameters import apply_trainable_patterns

        apply_trainable_patterns(self._model, cfg.trainable_params)
        self._loss_fn = nn.MSELoss()

    @classmethod
    def from_config(cls, cfg: BackendConfig, device: torch.device) -> MockBackend:
        return cls(cfg, device)

    # ----- training primitives -----

    def build_dataloader(self, data_cfg: DataConfig, partition_indices: list[int] | None) -> Iterator[dict[str, Any]]:
        return _synthetic_iterator(
            batch_size=data_cfg.batch_size,
            in_dim=self.IN_DIM,
            out_dim=self.OUT_DIM,
            seed=data_cfg.seed,
            n_samples=self.SYNTHETIC_DATASET_SIZE,
            indices=partition_indices,
        )

    @staticmethod
    def list_episodes(data_cfg: DataConfig) -> list[EpisodeMeta]:
        """Return synthetic episode metadata for deterministic partitioning.

        A staticmethod so it is callable from the CLASS with `data_cfg` alone
        — partitioning runs before any backend is instantiated. Deterministic
        grid of `n_tasks * eps_per_task` episodes (from `data_cfg.extra`,
        defaults 4 * 8 = 32). Episode indices double as sample indices into
        `build_dataloader`'s synthetic pool, so the universe must fit inside
        SYNTHETIC_DATASET_SIZE.
        """
        extra = data_cfg.extra or {}
        n_tasks = int(extra.get("n_tasks", 4))
        eps_per_task = int(extra.get("eps_per_task", 8))
        if n_tasks * eps_per_task > MockBackend.SYNTHETIC_DATASET_SIZE:
            raise ValueError(
                f"n_tasks * eps_per_task = {n_tasks * eps_per_task} exceeds the synthetic "
                f"sample pool ({MockBackend.SYNTHETIC_DATASET_SIZE}); episode indices would "
                "not be valid partition_indices for build_dataloader."
            )
        return [
            EpisodeMeta(
                episode_index=t * eps_per_task + i,
                task_index=t,
                task_name=f"task_{t}",
            )
            for t in range(n_tasks)
            for i in range(eps_per_task)
        ]

    def compute_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        obs = batch["observation"].to(self.device)
        act = batch["action"].to(self.device)
        pred = self._model(obs)
        loss = self._loss_fn(pred, act)
        return {"loss": loss}

    def build_optimizer(self, optim_cfg: OptimConfig) -> tuple[torch.optim.Optimizer, object | None]:
        return make_optimizer(self._model, optim_cfg)

    # ----- parameter exchange -----

    def get_state(self, scope: str = "shared") -> dict[str, torch.Tensor]:
        return trainable_state(self._model, self.cfg.shared_params, scope)

    def set_state(self, state: dict[str, torch.Tensor], scope: str = "shared") -> None:
        set_model_state(self._model, state)

    def load_checkpoint(self, path: str) -> None:
        load_model_checkpoint(self._model, path, self.device, "mock model")

    # ----- inference -----

    def build_policy(self):
        from src.adapters._mock.policy import MockPolicy

        return MockPolicy(self._model, in_dim=self.IN_DIM, device=self.device)

    # ----- metadata -----

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def name(self) -> str:
        return "mock"
