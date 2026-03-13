# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.recipe.sim_env import SimEnv


def test_sim_env_validation():
    # Test with valid inputs
    env = SimEnv(num_clients=3, clients=["client1", "client2", "client3"])
    assert env.num_clients == 3
    assert env.clients == ["client1", "client2", "client3"]

    # Test with inconsistent number of clients
    with pytest.raises(ValueError, match="Inconsistent number of clients"):
        SimEnv(num_clients=2, clients=["client1", "client2", "client3"])

    # Test with no clients specified (invalid)
    with pytest.raises(ValueError, match="Either 'num_clients' must be > 0 or 'clients' list must be provided"):
        SimEnv()

    # Test with empty clients list and zero num_clients (invalid)
    with pytest.raises(ValueError, match="Either 'num_clients' must be > 0 or 'clients' list must be provided"):
        SimEnv(num_clients=0, clients=[])

    # BUG-3 regression: when clients list is provided and num_clients=0,
    # env should derive client/thread counts from the list.
    env = SimEnv(num_clients=0, clients=["client1", "client2", "client3"])
    assert env.num_clients == 3
    assert env.num_threads == 3
