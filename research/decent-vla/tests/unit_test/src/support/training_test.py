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

"""cycle_loader must raise on a loader with no batches, which it would otherwise restart forever."""

from __future__ import annotations

import pytest

from src.support.training import cycle_loader


class _EmptyLoader:
    """A loader with no batches. It fails on a second epoch, so that a regression fails instead of hanging the test."""

    def __init__(self):
        self.epochs = 0

    def __iter__(self):
        self.epochs += 1
        assert self.epochs == 1, "cycle_loader restarted a loader that yields no batches"
        return iter(())


def test_loader_without_batches_raises():
    with pytest.raises(ValueError, match="yielded no batches"):
        next(cycle_loader(_EmptyLoader()))


def test_loader_with_batches_restarts_at_each_epoch():
    loader = cycle_loader([1, 2])

    assert [next(loader) for _ in range(5)] == [1, 2, 1, 2, 1]
