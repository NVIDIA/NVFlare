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

"""Policy/benchmark construction and deterministic site partitioning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import RunConfig
    from src.support.partitioning import PartitionManifest


def build_backend_factory(cfg: RunConfig, device: torch.device):
    if cfg.backend.name == "mock":
        from src.adapters._mock.backend import MockBackend as BackendClass
    elif cfg.backend.name == "lerobot":
        from src.adapters.lerobot.backend import LeRobotBackend as BackendClass
    elif cfg.backend.name == "groot":
        from src.adapters.groot.backend import GrootBackend as BackendClass
    else:
        raise ValueError(f"Unknown backend: {cfg.backend.name}")

    def factory():
        if cfg.backend.name == "lerobot":
            return BackendClass.from_config(cfg.backend, device, data_cfg=cfg.data)
        return BackendClass.from_config(cfg.backend, device)

    return factory, BackendClass


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def partition_for(cfg: RunConfig, backend_cls) -> tuple[list[list[int]], PartitionManifest]:
    """Build deterministic partitions from backend episode metadata or assigned repos."""
    if cfg.data.partitioning == "explicit_repos":
        from src.support.partitioning import PartitionManifest

        repos = cfg.data.client_repo_ids
        if not repos:
            raise ValueError(
                "partitioning='explicit_repos' requires data.client_repo_ids "
                "(a non-empty list of per-client repo lists)"
            )
        chunks = [[i] for i in range(len(repos))]
        manifest = PartitionManifest(
            scheme="explicit_repos",
            num_clients=len(repos),
            seed=cfg.data.seed,
            dataset_repo_id=None,
            dataset_revision=None,
            held_out_task_indices=[],
            held_out_task_names=[],
            clients={f"c{i}": {"repo_ids": list(repos[i])} for i in range(len(repos))},
            total_episodes=0,
        )
        return chunks, manifest

    from src.support.partitioning import load_heldout_json, partition_episodes

    episodes = backend_cls.list_episodes(cfg.data)
    held = load_heldout_json(cfg.data.held_out) if cfg.data.held_out else None
    return partition_episodes(
        episodes,
        scheme=cfg.data.partitioning,
        num_clients=cfg.data.num_clients,
        seed=cfg.data.seed,
        held_out=held,
        dataset_repo_id=cfg.data.repo_id,
        dataset_revision=None,
        client_task_indices=cfg.data.client_task_indices,
    )
