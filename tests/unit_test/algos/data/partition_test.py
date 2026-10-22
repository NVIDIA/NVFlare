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

"""Unit tests for the data partitioning utilities."""

import numpy as np
import pytest

from nvflare.algos.data.partition import (
    PartitionResult,
    dirichlet_label_partition,
    iid_partition,
    quantity_skew_partition,
)

NUM_CLASSES = 10
NUM_SAMPLES = 5000


@pytest.fixture
def labels():
    return np.repeat(np.arange(NUM_CLASSES), NUM_SAMPLES // NUM_CLASSES)


def _assert_valid_partition(result, num_samples, num_sites):
    assert result.num_sites == num_sites
    assert sorted(result.site_indices) == list(range(num_sites))
    all_indices = np.concatenate(list(result.site_indices.values()))
    # every sample is assigned to exactly one site
    assert np.array_equal(np.sort(all_indices), np.arange(num_samples))


def _assert_same(a, b):
    assert a.site_indices.keys() == b.site_indices.keys()
    for site in a.site_indices:
        assert np.array_equal(a.site_indices[site], b.site_indices[site])
    assert a.class_counts == b.class_counts


class TestDirichletLabelPartition:
    def test_covers_all_samples_once(self, labels):
        result = dirichlet_label_partition(labels, num_sites=8, alpha=0.5, seed=0)
        _assert_valid_partition(result, NUM_SAMPLES, 8)

    def test_deterministic_for_same_seed(self, labels):
        a = dirichlet_label_partition(labels, num_sites=5, alpha=0.5, seed=7)
        b = dirichlet_label_partition(labels, num_sites=5, alpha=0.5, seed=7)
        _assert_same(a, b)

    def test_different_seed_differs(self, labels):
        a = dirichlet_label_partition(labels, num_sites=5, alpha=0.5, seed=1)
        b = dirichlet_label_partition(labels, num_sites=5, alpha=0.5, seed=2)
        assert not all(np.array_equal(a.site_indices[s], b.site_indices[s]) for s in a.site_indices)

    def test_does_not_touch_global_rng(self, labels):
        np.random.seed(123)
        expected = np.random.rand(3)
        np.random.seed(123)
        dirichlet_label_partition(labels, num_sites=4, alpha=0.5, seed=0)
        assert np.array_equal(np.random.rand(3), expected)

    def test_class_counts_match_indices(self, labels):
        result = dirichlet_label_partition(labels, num_sites=4, alpha=0.5, seed=0)
        for site, idx in result.site_indices.items():
            assert sum(result.class_counts[site].values()) == len(idx)
            for label, count in result.class_counts[site].items():
                assert np.sum(labels[idx] == label) == count

    def test_small_alpha_is_more_skewed_than_large_alpha(self, labels):
        def max_class_share(alpha):
            result = dirichlet_label_partition(labels, num_sites=5, alpha=alpha, seed=0)
            shares = [max(c.values()) / sum(c.values()) for c in result.class_counts.values()]
            return np.mean(shares)

        assert max_class_share(0.1) > max_class_share(100.0)

    def test_min_size_retry(self, labels):
        # with a very small alpha a single draw regularly leaves a site nearly empty, so the retry loop must kick in
        result = dirichlet_label_partition(labels, num_sites=10, alpha=0.05, seed=3, min_size=100)
        assert min(len(idx) for idx in result.site_indices.values()) >= 100

    def test_min_size_unreachable_raises(self, labels):
        with pytest.raises(ValueError, match="could not give every site"):
            dirichlet_label_partition(
                labels, num_sites=10, alpha=0.01, seed=0, min_size=NUM_SAMPLES // 10, max_attempts=3
            )

    def test_non_contiguous_labels(self):
        labels = np.array([3, 7, 42] * 100)
        result = dirichlet_label_partition(labels, num_sites=3, alpha=1.0, seed=0, min_size=1)
        _assert_valid_partition(result, 300, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_sites": 0, "alpha": 0.5},
            {"num_sites": 2, "alpha": 0.0},
            {"num_sites": 2, "alpha": -1.0},
            {"num_sites": 2, "alpha": 0.5, "min_size": -1},
            {"num_sites": 2, "alpha": 0.5, "min_size": 10_000},
            {"num_sites": 2, "alpha": 0.5, "max_attempts": 0},
            {"num_sites": 2, "alpha": 0.5, "max_attempts": 2.5},
            {"num_sites": 2, "alpha": 0.5, "max_attempts": float("nan")},
            {"num_sites": 2, "alpha": 0.5, "min_size": 1.5},
            {"num_sites": 2, "alpha": 0.5, "min_size": float("nan")},
        ],
    )
    def test_invalid_arguments(self, labels, kwargs):
        with pytest.raises(ValueError):
            dirichlet_label_partition(labels, seed=0, **kwargs)

    def test_labels_must_be_integers(self):
        with pytest.raises(ValueError, match="integer"):
            dirichlet_label_partition(np.array(["cat", "dog"] * 50), num_sites=2, alpha=0.5, seed=0, min_size=0)

    def test_labels_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            dirichlet_label_partition(np.zeros((4, 2)), num_sites=2, alpha=0.5, seed=0, min_size=0)

    def test_single_site(self, labels):
        result = dirichlet_label_partition(labels, num_sites=1, alpha=0.5, seed=0)
        _assert_valid_partition(result, NUM_SAMPLES, 1)


class TestIidPartition:
    def test_without_labels(self):
        result = iid_partition(1003, num_sites=4, seed=0)
        _assert_valid_partition(result, 1003, 4)
        sizes = [len(idx) for idx in result.site_indices.values()]
        assert max(sizes) - min(sizes) <= 1
        assert result.class_counts == {}

    def test_deterministic(self):
        _assert_same(iid_partition(500, 4, seed=3), iid_partition(500, 4, seed=3))

    def test_stratified_class_balance(self, labels):
        result = iid_partition(NUM_SAMPLES, num_sites=7, seed=0, labels=labels)
        _assert_valid_partition(result, NUM_SAMPLES, 7)
        for label in range(NUM_CLASSES):
            counts = [c.get(label, 0) for c in result.class_counts.values()]
            assert max(counts) - min(counts) <= 1

    def test_stratified_site_sizes_are_balanced_with_uneven_classes(self):
        labels = np.array([0] * 11 + [1] * 11 + [2] * 11)
        result = iid_partition(33, num_sites=4, seed=0, labels=labels)
        sizes = [len(idx) for idx in result.site_indices.values()]
        assert max(sizes) - min(sizes) <= 1

    def test_label_length_mismatch(self, labels):
        with pytest.raises(ValueError, match="num_samples"):
            iid_partition(NUM_SAMPLES + 1, num_sites=2, seed=0, labels=labels)

    def test_more_sites_than_samples(self):
        with pytest.raises(ValueError):
            iid_partition(3, num_sites=4, seed=0)

    def test_invalid_num_sites(self):
        with pytest.raises(ValueError):
            iid_partition(10, num_sites=0, seed=0)


class TestQuantitySkewPartition:
    def test_covers_all_samples_once(self):
        result = quantity_skew_partition(1000, num_sites=5, seed=0)
        _assert_valid_partition(result, 1000, 5)

    def test_deterministic(self):
        _assert_same(quantity_skew_partition(1000, 5, seed=4), quantity_skew_partition(1000, 5, seed=4))

    def test_sizes_are_unequal_for_small_alpha(self):
        result = quantity_skew_partition(10_000, num_sites=5, seed=0, alpha=0.3, min_size=10)
        sizes = [len(idx) for idx in result.site_indices.values()]
        assert min(sizes) >= 10
        assert max(sizes) > 2 * min(sizes)

    def test_min_size_retry(self):
        result = quantity_skew_partition(1000, num_sites=5, seed=1, alpha=1.0, min_size=100)
        assert min(len(idx) for idx in result.site_indices.values()) >= 100

    def test_min_size_unreachable_raises(self):
        with pytest.raises(ValueError, match="could not give every site"):
            quantity_skew_partition(1000, num_sites=10, seed=0, alpha=0.01, min_size=100, max_attempts=3)

    def test_class_counts_with_labels(self, labels):
        result = quantity_skew_partition(NUM_SAMPLES, num_sites=4, seed=0, labels=labels)
        assert sum(sum(c.values()) for c in result.class_counts.values()) == NUM_SAMPLES

    def test_label_length_mismatch(self, labels):
        with pytest.raises(ValueError, match="num_samples"):
            quantity_skew_partition(10, num_sites=2, seed=0, labels=labels)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            quantity_skew_partition(100, num_sites=2, seed=0, alpha=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_attempts": 2.5},
            {"max_attempts": float("nan")},
            {"min_size": 1.5},
            {"min_size": float("nan")},
        ],
    )
    def test_non_integer_min_size_and_max_attempts(self, kwargs):
        with pytest.raises(ValueError):
            quantity_skew_partition(100, num_sites=2, seed=0, **kwargs)


class TestPartitionResultJson:
    def test_round_trip(self, labels, tmp_path):
        result = dirichlet_label_partition(labels, num_sites=4, alpha=0.5, seed=0)
        path = str(tmp_path / "nested" / "split.json")
        result.save(path)
        loaded = PartitionResult.load(path)
        _assert_same(result, loaded)
        assert all(isinstance(site, int) for site in loaded.site_indices)
        assert all(isinstance(label, int) for counts in loaded.class_counts.values() for label in counts)

    def test_round_trip_without_class_counts(self, tmp_path):
        result = iid_partition(100, num_sites=3, seed=0)
        path = str(tmp_path / "split.json")
        result.save(path)
        _assert_same(result, PartitionResult.load(path))
