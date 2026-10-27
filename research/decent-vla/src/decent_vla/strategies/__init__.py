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

"""Built-in FL strategies.

v0.1 ships FedAvg, FedAdam, FedProx. Each is a single file (~40-80 lines)
and subclasses `StrategyBase` from `decent_vla.core.strategy`.
"""
from __future__ import annotations

from decent_vla.strategies.fedadam import FedAdam
from decent_vla.strategies.fedavg import FedAvg
from decent_vla.strategies.fedprox import FedProx

__all__ = ["FedAvg", "FedAdam", "FedProx"]
