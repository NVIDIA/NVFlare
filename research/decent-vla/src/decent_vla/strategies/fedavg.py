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

"""FedAvg — the canonical aggregation-only strategy.

Reference: McMahan et al. 2017, "Communication-Efficient Learning of Deep
Networks from Decentralized Data".

The base class `StrategyBase` already provides FedAvg semantics in its
defaults (weighted aggregate, delta = current - global). FedAvg here is
just a subclass with a name; we ship it as its own file so users can
read the canonical implementation.
"""
from __future__ import annotations

from decent_vla.core.strategy import StrategyBase


class FedAvg(StrategyBase):
    name = "fedavg"

    def __init__(self, cfg=None):
        # No state — FedAvg uses base defaults.
        self.cfg = cfg
