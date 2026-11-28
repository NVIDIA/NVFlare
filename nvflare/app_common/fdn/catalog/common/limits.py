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

"""Bounds for a federated catalog read."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadLimits:
    max_duration_s: float
    max_calls: int
    max_records: int


DEFAULT_LIMITS = {"max_duration_s": 60, "max_calls": 1000, "max_records": 10000}
