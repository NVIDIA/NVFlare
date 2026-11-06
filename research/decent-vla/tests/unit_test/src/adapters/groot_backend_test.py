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

"""The GR00T adapter's shared normalization statistics and its check for a sampler with no batches.

No Isaac-GR00T, model or video: stand-ins replace the upstream dataset classes, while the adapter's episode mapping,
flattening and shard-grouped sampling run for real. The stand-ins show how the adapter uses upstream, not that
upstream's merge is right.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from src.adapters.groot.backend import GrootBackend
from src.config import BackendConfig, DataConfig

SHARD_LENGTH = 4
EPISODES_PER_SUITE = 2


class _Sharded:
    """Stand-in for upstream's ShardedSingleStepDataset, with one shard of SHARD_LENGTH timesteps."""

    def __init__(self, path, episodes):
        self.path = path
        self.episodes = episodes
        self.processor = None  # the adapter attaches the processor only to the datasets it trains on

    def __len__(self):
        return 1

    def get_shard_length(self, shard_idx):
        return SHARD_LENGTH

    def get_shard(self, shard_idx):
        return list(range(SHARD_LENGTH))

    def set_processor(self, processor):
        self.processor = processor


class _Mixture:
    """Stand-in for upstream's ShardedMixtureDataset, which sets the merged statistics of its datasets on the processor.

    The "statistics" here record which datasets went into the merge.
    """

    def __init__(self, datasets, weights, processor, seed, training, override_pretraining_statistics):
        assert override_pretraining_statistics, "upstream would keep statistics already on the processor"
        processor.statistics = [(dataset.path, dataset.episodes) for dataset in datasets]


@pytest.fixture
def built(monkeypatch):
    """Replace the upstream classes the adapter calls; returns the list of every dataset the adapter builds."""
    datasets = []

    def build_partitioned_dataset(dataset_path, keep_episode_indices, **kwargs):
        datasets.append(_Sharded(dataset_path, keep_episode_indices))
        return datasets[-1]

    monkeypatch.setattr("src.adapters.groot._dataset.build_partitioned_dataset", build_partitioned_dataset)
    monkeypatch.setitem(
        sys.modules, "gr00t.data.dataset.sharded_mixture_dataset", SimpleNamespace(ShardedMixtureDataset=_Mixture)
    )
    monkeypatch.setattr("src.adapters.groot.policy.GrootPolicy", lambda *args, **kwargs: None)
    return datasets


@pytest.fixture
def suites(tmp_path):
    """Three suite folders. Global episode indices: suite 0 has 0 and 1, suite 1 has 2 and 3, suite 2 has 4 and 5."""
    paths = []
    for name in ("suite0", "suite1", "suite2"):
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        episodes = [{"episode_index": i, "tasks": [f"{name} task {i}"]} for i in range(EPISODES_PER_SUITE)]
        (meta / "episodes.jsonl").write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
        ranges = {"x": {"start": 0, "end": 1}}
        (meta / "modality.json").write_text(json.dumps({"state": ranges, "action": ranges}))
        paths.append(str(tmp_path / name))
    return paths


def _backend(**extra):
    backend = GrootBackend(BackendConfig(name="groot", extra=extra), torch.device("cpu"))
    backend._processor = SimpleNamespace(collator=lambda batch: batch, statistics=None)
    return backend


def _data(suites, batch_size=2, **extra):
    return DataConfig(batch_size=batch_size, extra={"dataset_paths": suites, **extra})


@pytest.mark.usefixtures("built")
def test_clients_and_evaluation_use_statistics_from_every_suite(suites):
    every_suite = [(path, None) for path in suites]

    for partition in ([0, 1], [2, 5], None):  # one suite, two suites, everything
        client = _backend()
        client.build_dataloader(_data(suites), partition)
        assert client._processor.statistics == every_suite, partition

    evaluation = _backend(eval_dataset_paths=suites)
    evaluation.build_policy()
    assert evaluation._processor.statistics == every_suite


def test_training_data_stays_partitioned(suites, built):
    _backend().build_dataloader(_data(suites), [1, 4])

    training = [(dataset.path, dataset.episodes) for dataset in built if dataset.processor is not None]
    assert training == [(suites[0], [1]), (suites[2], [0])]  # global episodes mapped to each suite's own indices


@pytest.mark.usefixtures("built")
def test_shard_grouped_sampler_without_batches_is_rejected(suites):
    data = _data(suites, batch_size=SHARD_LENGTH + 1, shard_grouped_batches=True, drop_last=True)

    with pytest.raises(ValueError, match="yields no batches"):
        _backend().build_dataloader(data, [0, 2])


@pytest.mark.usefixtures("built")
@pytest.mark.parametrize("batch_size, drop_last", [(SHARD_LENGTH, True), (SHARD_LENGTH + 1, False)])
def test_shard_grouped_sampler_with_batches_is_accepted(suites, batch_size, drop_last):
    data = _data(suites, batch_size=batch_size, shard_grouped_batches=True, drop_last=drop_last)

    batch = next(_backend().build_dataloader(data, [0, 2]))

    assert len(batch) == SHARD_LENGTH
