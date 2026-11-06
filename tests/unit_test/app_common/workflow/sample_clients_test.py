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

import random
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.workflows.fedavg import FedAvg

NAMES = ["site-1", "site-2", "site-3", "site-4", "site-5", "site-6"]


def make_controller(names, seed=None):
    controller = FedAvg(num_clients=len(names), num_rounds=1, sampling_seed=seed)
    controller.fl_ctx = FLContext()
    controller.engine = SimpleNamespace(get_clients=lambda: [SimpleNamespace(name=n) for n in names])
    return controller


def schedule(controller, num_clients, rounds):
    return [controller.sample_clients(num_clients) for _ in range(rounds)]


class TestSampleClients:
    def test_unseeded_returns_a_subset_and_all_when_enough(self):
        controller = make_controller(NAMES)
        picked = controller.sample_clients(2)
        assert len(picked) == 2 and set(picked) <= set(NAMES)
        assert controller.sample_clients(len(NAMES)) == NAMES
        assert controller.sample_clients(None) == NAMES

    def test_seeded_schedule_repeats_for_the_same_available_clients(self):
        first = make_controller(NAMES, seed=7)
        second = make_controller(list(reversed(NAMES)), seed=7)  # same clients, different engine order
        assert schedule(first, 3, 10) == schedule(second, 3, 10)

    def test_seeded_schedule_changes_with_the_seed(self):
        assert schedule(make_controller(NAMES, seed=7), 3, 20) != schedule(make_controller(NAMES, seed=8), 3, 20)

    def test_seeded_sampling_does_not_touch_global_random_state(self):
        original_state = random.getstate()
        try:
            random.seed(1234)
            expected = [random.random() for _ in range(3)]
            random.seed(1234)
            schedule(make_controller(NAMES, seed=7), 3, 5)
            assert [random.random() for _ in range(3)] == expected
        finally:
            random.setstate(original_state)

    def test_seeded_full_participation_keeps_engine_order(self):
        controller = make_controller(NAMES, seed=7)
        assert controller.sample_clients(len(NAMES)) == NAMES

    @pytest.mark.parametrize("seed", [-1, "7", 1.5])
    def test_invalid_seed_is_rejected(self, seed):
        with pytest.raises((TypeError, ValueError)):
            make_controller(NAMES, seed=seed)
