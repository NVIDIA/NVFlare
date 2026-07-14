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

"""
Regression tests for ccwf_job config dataclass defaults.

SwarmServerConfig.min_clients was previously defaulted to None, which the
add_swarm() wiring forwarded to SwarmServerController.__init__ (typed
min_clients: int = 0). The value then reached BaseServerCtl.__init__ where
`if min_clients < 0:` raised TypeError ("'<' not supported between instances
of 'NoneType' and 'int'") at construction time, making the swarm CIFAR10
example fail before any simulator was even started.

The fix sets SwarmServerConfig.min_clients default to 0 so that the type
contract (int) is honored end-to-end and the documented "0 means all
participating clients are required" semantics from BaseServerCtl applies.
"""

from nvflare.app_common.ccwf.ccwf_job import SwarmServerConfig


def test_swarm_server_config_default_min_clients_is_zero():
    """Default SwarmServerConfig must use min_clients=0, not None.

    None as a default would propagate through add_swarm() into
    BaseServerCtl.__init__ where `if min_clients < 0:` raises TypeError.
    """
    config = SwarmServerConfig(num_rounds=1)
    assert config.min_clients == 0
    assert config.min_clients is not None


def test_swarm_server_config_explicit_min_clients_preserved():
    """Explicit min_clients value passes through unchanged."""
    config = SwarmServerConfig(num_rounds=1, min_clients=2)
    assert config.min_clients == 2
