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

"""Plugin builders + generic partitioning, shared by the CLI and runners.

Extracted from `cli/main.py` so the NVFlare entrypoints (and anything else
that is not the argparse CLI) can construct backends/strategies/benchmarks
and partition data from a `RunConfig`. Lookup failures raise `SystemExit`
with actionable messages — identical to the original CLI behavior.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from decent_vla.core.registry import discover

if TYPE_CHECKING:
    from decent_vla.configs.schemas import EvalConfig, RunConfig
    from decent_vla.core.partitioner import PartitionManifest


def build_backend_factory(cfg: RunConfig, device: torch.device):
    reg = discover()
    if cfg.backend.name not in reg.backends:
        raise SystemExit(f"Backend {cfg.backend.name!r} not installed. " f"Available: {sorted(reg.backends)}")
    cls = reg.backends[cfg.backend.name]

    def factory():
        dev = device
        # Only LeRobotBackend.from_config accepts data_cfg (explicit_repos
        # shared normalization stats) — other backends (mock, groot) don't
        # take this kwarg, so pass it only when supported.
        if "data_cfg" in inspect.signature(cls.from_config).parameters:
            return cls.from_config(cfg.backend, dev, data_cfg=cfg.data)
        return cls.from_config(cfg.backend, dev)

    return factory, cls


def build_benchmark(cfg_eval: EvalConfig):
    reg = discover()
    bench_name = cfg_eval.benchmark.name
    if bench_name not in reg.benchmarks:
        raise SystemExit(f"Benchmark {bench_name!r} not installed. Available: {sorted(reg.benchmarks)}")
    cls = reg.benchmarks[bench_name]
    return cls.from_config(cfg_eval.benchmark)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def partition_for(cfg: RunConfig, backend_cls) -> tuple[list[list[int]], PartitionManifest]:
    """Generic partitioning for ANY backend: prefers the backend's optional
    `list_episodes` metadata; falls back to a flat sample universe.

    `list_episodes` is looked up on the CLASS, so it must be a staticmethod
    or classmethod callable with `data_cfg` alone (see core/backend.py) —
    partitioning happens before any backend is instantiated.
    """
    if cfg.data.partitioning == "explicit_repos":
        from decent_vla.core.partitioner import PartitionManifest

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

    from decent_vla.core.partitioner import EpisodeMeta, load_heldout_json, partition_episodes

    list_eps = getattr(backend_cls, "list_episodes", None)
    if list_eps is None:
        n = (cfg.data.extra or {}).get("num_samples", 256)
        episodes = [EpisodeMeta(i, 0, "samples") for i in range(n)]
    else:
        episodes = list_eps(cfg.data)
    held = load_heldout_json(cfg.data.held_out) if cfg.data.held_out else None
    return partition_episodes(
        episodes,
        scheme=cfg.data.partitioning,
        num_clients=cfg.data.num_clients,
        seed=cfg.data.seed,
        held_out=held,
        dataset_repo_id=cfg.data.repo_id,
        dataset_revision=cfg.data.revision,
        client_task_indices=cfg.data.client_task_indices,
    )
