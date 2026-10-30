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

"""What the shard-grouped batch sampler must not break.

The sampler exists for speed — the flattener decodes a whole shard to return one
timestep, so a global shuffle re-decodes ~29 shards in 30 — but speed is the easy
part. The properties worth pinning are the ones a faster sampler could silently
lose: every sample still seen once per epoch, every batch confined to one shard
(otherwise the shard load buys nothing), and the same seed replaying the same
batches so a resumed round is the round it claims to be.

No GR00T checkout, no video: `_shard_groups` reads `_index`, and the sampler
yields plain integers, so stand-ins with the right `_index` shape exercise the
real logic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import ConcatDataset  # noqa: E402

from decent_vla.adapters.groot.backend import (  # noqa: E402
    _shard_groups,
    _ShardGroupedBatchSampler,
)


class _FakeFlat:
    """Stand-in exposing only what `_shard_groups` reads: `_index` and a length."""

    def __init__(self, index):
        self._index = list(index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i):  # ConcatDataset requires indexability
        return self._index[i]


def _index_for(shard_lengths):
    """`[(shard_idx, offset), ...]` for consecutive shards of the given lengths."""
    return [(s, o) for s, n in enumerate(shard_lengths) for o in range(n)]


def test_shard_groups_partition_a_single_dataset_by_shard():
    groups = _shard_groups(_FakeFlat(_index_for([3, 2])))

    assert groups == [[0, 1, 2], [3, 4]]


def test_shard_groups_offset_each_concat_member_into_global_indices():
    combined = ConcatDataset([_FakeFlat(_index_for([2, 1])), _FakeFlat(_index_for([2]))])

    groups = _shard_groups(combined)

    # Second member's shard 0 is a group of its own, numbered from 3 — two
    # suites' shard 0 must never share a batch just because they share an index.
    assert groups == [[0, 1], [2], [3, 4]]


def test_shard_groups_are_ordered_by_shard_index():
    # Interleaved shards: grouping must key on shard, not on arrival order.
    groups = _shard_groups(_FakeFlat([(1, 0), (0, 0), (1, 1), (0, 1)]))

    assert groups == [[1, 3], [0, 2]]


def test_every_index_appears_exactly_once_per_epoch():
    groups = _shard_groups(_FakeFlat(_index_for([7, 5, 4])))
    sampler = _ShardGroupedBatchSampler(groups, batch_size=3, seed=0)

    drawn = [i for batch in sampler for i in batch]

    assert sorted(drawn) == list(range(16))


def test_no_batch_straddles_two_shards():
    groups = _shard_groups(_FakeFlat(_index_for([7, 5, 4])))
    owner = {i: g for g, idx in enumerate(groups) for i in idx}
    sampler = _ShardGroupedBatchSampler(groups, batch_size=3, seed=0)

    # One shard per batch is the whole point: a straddling batch would force a
    # second shard decode and give back the speedup the sampler was written for.
    assert all(len({owner[i] for i in batch}) == 1 for batch in sampler)


def test_len_matches_the_batches_actually_yielded():
    groups = _shard_groups(_FakeFlat(_index_for([7, 5, 4])))

    for drop_last, expected in ((False, 3 + 2 + 2), (True, 2 + 1 + 1)):
        sampler = _ShardGroupedBatchSampler(groups, batch_size=3, drop_last=drop_last)
        assert len(sampler) == expected == len(list(sampler))


def test_drop_last_keeps_only_full_batches():
    sampler = _ShardGroupedBatchSampler([[0, 1, 2, 3, 4]], batch_size=2, drop_last=True)

    assert [len(b) for b in sampler] == [2, 2]


def test_same_seed_replays_the_same_epochs():
    groups = _shard_groups(_FakeFlat(_index_for([6, 6])))
    a = _ShardGroupedBatchSampler(groups, batch_size=4, seed=7)
    b = _ShardGroupedBatchSampler(groups, batch_size=4, seed=7)

    # Two epochs each: the per-epoch redraw must be a function of the seed, or a
    # resumed run silently trains on a different order than the one it reports.
    assert [list(a), list(a)] == [list(b), list(b)]


def test_successive_epochs_redraw_the_order():
    groups = _shard_groups(_FakeFlat(_index_for([6, 6])))
    sampler = _ShardGroupedBatchSampler(groups, batch_size=4, seed=7)

    assert list(sampler) != list(sampler)


def test_empty_groups_are_dropped():
    sampler = _ShardGroupedBatchSampler([[], [0, 1], []], batch_size=2)

    assert len(sampler) == 1
    assert list(sampler) == [[0, 1]]


def test_non_positive_batch_size_is_refused():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _ShardGroupedBatchSampler([[0, 1]], batch_size=0)
