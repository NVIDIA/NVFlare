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

"""explicit_repos concat loader wiring — fakes lerobot make_dataset/sampler."""
from __future__ import annotations

import sys
import types

import torch


class _FakeDS(torch.utils.data.Dataset):
    def __init__(self, n, n_eps):
        self._n = n
        self.meta = types.SimpleNamespace(
            total_episodes=n_eps,
            episodes={"dataset_from_index": list(range(n_eps)),
                      "dataset_to_index": list(range(1, n_eps + 1))},
        )
    def __len__(self): return self._n
    def __getitem__(self, i): return {"index": i}


def _install_fakes(monkeypatch, sizes):
    made = iter(sizes)
    calls = {"repos": []}

    def fake_make_dataset(tp_cfg):
        calls["repos"].append(tp_cfg.dataset.repo_id)
        n, n_eps = next(made)
        return _FakeDS(n, n_eps)

    factory_mod = types.ModuleType("lerobot.datasets.factory")
    factory_mod.make_dataset = fake_make_dataset
    sampler_mod = types.ModuleType("lerobot.datasets.sampler")

    class _FakeSampler:
        def __init__(self, frm, to, episode_indices_to_use, drop_n_last_frames, shuffle):
            self._idx = [frm[e] for e in episode_indices_to_use]
        def __iter__(self): return iter(self._idx)
        def __len__(self): return len(self._idx)

    sampler_mod.EpisodeAwareSampler = _FakeSampler
    monkeypatch.setitem(sys.modules, "lerobot.datasets.factory", factory_mod)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.sampler", sampler_mod)
    return calls


def _make_backend():
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    b = LeRobotBackend.__new__(LeRobotBackend)
    b.device = torch.device("cpu")
    b._train_pipeline_cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(repo_id="orig/repo", root=None),
        policy=types.SimpleNamespace(drop_n_last_frames=0),
    )
    return b


def test_concat_loader_visits_all_client_repos(monkeypatch):
    from decent_vla.configs.schemas import DataConfig
    calls = _install_fakes(monkeypatch, sizes=[(10, 2), (20, 4)])
    b = _make_backend()
    data_cfg = DataConfig(name="lerobot", partitioning="explicit_repos",
                          client_repo_ids=[["r/a", "r/b"], ["r/c"]],
                          batch_size=2, num_workers=0)
    it = b.build_dataloader(data_cfg, [0])
    assert calls["repos"] == ["r/a", "r/b"]
    assert b._explicit_frame_count == 6
    batch = next(it)
    assert "index" in batch


def test_concat_loader_offsets_second_dataset(monkeypatch):
    from decent_vla.configs.schemas import DataConfig
    _install_fakes(monkeypatch, sizes=[(10, 2), (20, 3)])
    b = _make_backend()
    data_cfg = DataConfig(name="lerobot", partitioning="explicit_repos",
                          client_repo_ids=[["r/a", "r/b"]], batch_size=4, num_workers=0)
    b.build_dataloader(data_cfg, [0])
    assert b._explicit_combined_idx == [0, 1, 10, 11, 12]


def test_explicit_frame_count_available_for_weighting(monkeypatch):
    from decent_vla.configs.schemas import DataConfig
    _install_fakes(monkeypatch, sizes=[(10, 2), (20, 4)])
    b = _make_backend()
    dc = DataConfig(name="lerobot", partitioning="explicit_repos",
                    client_repo_ids=[["r/a", "r/b"]], batch_size=2, num_workers=0)
    b.build_dataloader(dc, [0])
    assert b.num_train_frames() == 6
    # outside explicit_repos, accessor returns None
    b2 = _make_backend()
    assert b2.num_train_frames() is None
