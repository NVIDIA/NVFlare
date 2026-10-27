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

"""Partitioner types: held-out spec parsing, manifest hashing & IO."""
from __future__ import annotations

import dataclasses
import json
import random

import pytest

from decent_vla.core.partitioner import (
    EpisodeMeta,
    HeldOutSpec,
    PartitionManifest,
    load_heldout_json,
)

HELDOUT = {
    "description": "test held-out",
    "tasks": [
        {"suite": "libero_spatial", "dataset_task_index": 32, "eval_task_id": 5,
         "task_name": "pick up the black bowl on the ramekin and place it on the plate"},
        {"suite": "libero_goal", "dataset_task_index": 17, "eval_task_id": 1,
         "task_name": "put the bowl on the stove"},
    ],
}


def test_load_heldout_json(tmp_path):
    p = tmp_path / "heldout.json"
    p.write_text(json.dumps(HELDOUT))
    spec = load_heldout_json(p)
    assert isinstance(spec, HeldOutSpec)
    assert spec.dataset_task_indices == {32, 17}
    assert spec.eval_ids_for_suite("libero_spatial") == {5}
    assert spec.eval_ids_for_suite("libero_object") == set()


def test_manifest_roundtrip_and_hash(tmp_path):
    m = PartitionManifest(
        scheme="task_round_robin", num_clients=2, seed=42,
        dataset_repo_id="mock/ds", dataset_revision=None,
        held_out_task_indices=[2],
        held_out_task_names=["t2"],
        clients={"c0": {"task_names": ["t0"], "episode_indices": [0, 3]},
                 "c1": {"task_names": ["t1"], "episode_indices": [1, 4]}},
        total_episodes=4,
    )
    path = tmp_path / "partition_manifest.json"
    m.write(path)
    m2 = PartitionManifest.read(path)
    assert m2 == m
    assert m.content_hash() == m2.content_hash()
    assert m.content_hash().startswith("sha256:")
    # hash is order-insensitive over dict key insertion but sensitive to content
    m_reversed = PartitionManifest(
        scheme="task_round_robin", num_clients=2, seed=42,
        dataset_repo_id="mock/ds", dataset_revision=None,
        held_out_task_indices=[2],
        held_out_task_names=["t2"],
        clients={"c1": {"task_names": ["t1"], "episode_indices": [1, 4]},
                 "c0": {"task_names": ["t0"], "episode_indices": [0, 3]}},
        total_episodes=4,
    )
    assert m_reversed.content_hash() == m.content_hash()
    m3 = PartitionManifest.read(path)
    m3.clients["c0"]["episode_indices"] = [0]
    assert m3.content_hash() != m.content_hash()


def test_episode_meta_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        EpisodeMeta(0, 0, "t").task_index = 1


from decent_vla.core.partitioner import HeldOutTask  # noqa: E402
from decent_vla.runtime.partitioning import partition_episodes  # noqa: E402


def _episodes(n_tasks: int, eps_per_task: int) -> list[EpisodeMeta]:
    out, idx = [], 0
    for t in range(n_tasks):
        for _ in range(eps_per_task):
            out.append(EpisodeMeta(episode_index=idx, task_index=t, task_name=f"task_{t}"))
            idx += 1
    return out


def test_task_round_robin_with_heldout():
    eps = _episodes(n_tasks=40, eps_per_task=3)        # 120 episodes
    held = HeldOutSpec(tasks=tuple(
        HeldOutTask(suite="s", dataset_task_index=i, eval_task_id=0, task_name=f"task_{i}")
        for i in (32, 30, 27, 29, 17, 14, 3, 5)))
    chunks, manifest = partition_episodes(
        eps, scheme="task_round_robin", num_clients=8, seed=42, held_out=held,
        dataset_repo_id="mock/ds")
    assert len(chunks) == 8
    all_eps = sorted(e for c in chunks for e in c)
    assert len(all_eps) == 32 * 3                       # 8 tasks excluded
    assert len(set(all_eps)) == len(all_eps)            # no overlap
    held_eps = {e.episode_index for e in eps if e.task_index in held.dataset_task_indices}
    assert held_eps.isdisjoint(all_eps)                 # 0 held-out episodes anywhere
    for _cid, info in manifest.clients.items():
        assert len(info["task_names"]) == 4             # 32 tasks / 8 clients
        assert not set(info["task_names"]) & set(manifest.held_out_task_names)
    assert manifest.total_episodes == 96


def test_iid_episode_excludes_heldout_and_is_seeded():
    eps = _episodes(n_tasks=4, eps_per_task=10)
    held = HeldOutSpec(tasks=(HeldOutTask("s", 3, 0, "task_3"),))
    c1, m1 = partition_episodes(eps, "iid_episode", 2, seed=7, held_out=held)
    c2, _ = partition_episodes(eps, "iid_episode", 2, seed=7, held_out=held)
    assert c1 == c2                                     # deterministic by seed
    assert sum(len(c) for c in c1) == 30
    assert m1.scheme == "iid_episode"


def test_scheme_aliases_and_unknown():
    eps = _episodes(2, 4)
    chunks, _ = partition_episodes(eps, "task", 2, seed=1)      # alias
    assert len(chunks) == 2
    chunks, _ = partition_episodes(eps, "iid", 2, seed=1)       # alias
    assert len(chunks) == 2
    with pytest.raises(ValueError, match="Unknown partitioning scheme"):
        partition_episodes(eps, "bogus", 2, seed=1)


def test_heldout_unknown_task_index_raises():
    eps = _episodes(4, 2)
    held = HeldOutSpec(tasks=(HeldOutTask("s", 99, 0, "nonexistent"),))
    with pytest.raises(ValueError, match="absent from the dataset"):
        partition_episodes(eps, "task_round_robin", 2, seed=1, held_out=held)


def test_error_paths():
    eps = _episodes(2, 4)
    with pytest.raises(ValueError):
        partition_episodes(eps, "task_round_robin", 0, seed=1)
    eps3 = _episodes(2, 3)
    with pytest.raises(ValueError):
        partition_episodes(eps3, "task_round_robin", 3, seed=1)


def test_task_integrity():
    eps = _episodes(4, 6)
    chunks, manifest = partition_episodes(eps, "task_round_robin", 3, seed=42)
    task_index_by_ep = {e.episode_index: e.task_index for e in eps}
    task_index_by_name = {e.task_name: e.task_index for e in eps}
    for cid, info in manifest.clients.items():
        i = int(cid[1:])
        expected_task_indices = {task_index_by_name[n] for n in info["task_names"]}
        actual_task_indices = {task_index_by_ep[ep] for ep in chunks[i]}
        assert actual_task_indices == expected_task_indices, (
            f"client {cid}: episodes span tasks {actual_task_indices} "
            f"but manifest says {expected_task_indices}"
        )


def test_iid_episode_input_order_invariant():
    eps = _episodes(3, 5)
    shuffled = list(eps)
    random.Random(0).shuffle(shuffled)
    chunks_orig, _ = partition_episodes(eps, "iid_episode", 3, seed=7)
    chunks_shuffled, _ = partition_episodes(shuffled, "iid_episode", 3, seed=7)
    assert chunks_orig == chunks_shuffled


def test_load_heldout_8_config():
    """configs/libero_heldout_8.json — the held-out task spec for LIBERO."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "configs" / "libero_heldout_8.json"
    spec = load_heldout_json(path)
    assert len(spec.tasks) == 8
    suites = [t.suite for t in spec.tasks]
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        assert suites.count(suite) == 2, f"{suite}: expected 2 held-out tasks"
    assert spec.dataset_task_indices == {32, 30, 27, 29, 17, 14, 3, 5}


def test_mock_backend_list_episodes(backend, data_cfg):
    eps = backend.list_episodes(data_cfg)
    assert len(eps) >= data_cfg.num_clients
    assert all(isinstance(e, EpisodeMeta) for e in eps)
    assert eps == backend.list_episodes(data_cfg)   # deterministic
    # Must be callable from the CLASS with data_cfg alone (staticmethod):
    # partitioning runs before any backend is instantiated.
    from decent_vla.adapters._mock.backend import MockBackend

    assert MockBackend.list_episodes(data_cfg) == eps
    n = data_cfg.extra.get("n_tasks", 4) * data_cfg.extra.get("eps_per_task", 8)
    assert len(eps) == n
