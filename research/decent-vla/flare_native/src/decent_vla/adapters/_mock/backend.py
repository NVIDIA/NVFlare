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

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from decent_vla._helpers.checkpoints import CheckpointSpec
from decent_vla._helpers.partitioning import EpisodeMeta
from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig


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
        from decent_vla._helpers.parameters import apply_trainable_patterns

        apply_trainable_patterns(self._model, cfg.trainable_params)
        self._loss_fn = nn.MSELoss()
        self._seed = cfg.extra.get("data_seed", 42) if cfg.extra else 42

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
        opt = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
        return opt, None

    # ----- parameter exchange -----

    def get_state(self, scope: str = "shared") -> dict[str, torch.Tensor]:
        # Mirror the real adapters: "all" = every TRAINABLE param; "shared" =
        # the cfg.shared_params subset of it; "personal" = the complement.
        # Without patterns, shared == all and personal is empty (historical).
        trainable_names = {n for n, p in self._model.named_parameters() if p.requires_grad}
        trainable = {k: v.detach().clone() for k, v in self._model.state_dict().items() if k in trainable_names}
        if scope == "all":
            return trainable
        from decent_vla._helpers.parameters import resolve_shared_names

        shared = resolve_shared_names(trainable, self.cfg.shared_params)
        if scope == "personal":
            return {k: v for k, v in trainable.items() if k not in shared}
        return {k: v for k, v in trainable.items() if k in shared}

    def set_state(self, state: dict[str, torch.Tensor], scope: str = "shared") -> None:
        if scope == "personal":
            return  # nothing to set
        own = self._model.state_dict()
        casted = {k: v.to(own[k].dtype).to(own[k].device) for k, v in state.items() if k in own}
        own.update(casted)
        self._model.load_state_dict(own, strict=False)

    # ----- checkpointing -----

    def save_checkpoint(self, path: str, spec: CheckpointSpec) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "include": spec.include,
                "extra": spec.extra_state,
            },
            p,
        )
        # Also write a sidecar JSON so checkpoints are inspectable without torch.
        sidecar = p.with_suffix(p.suffix + ".meta.json")
        with sidecar.open("w") as f:
            json.dump({"include": spec.include, "name": self.name}, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self._model.load_state_dict(blob["state_dict"], strict=False)

    # ----- inference -----

    def build_policy(self):
        from decent_vla.adapters._mock.policy import MockPolicy

        return MockPolicy(self._model, in_dim=self.IN_DIM, out_dim=self.OUT_DIM, device=self.device)

    # ----- metadata -----

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_personal_params(self) -> bool:
        return False
