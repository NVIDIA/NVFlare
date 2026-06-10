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

"""Shared helpers for extracting benchmark record identity."""

from __future__ import annotations

from typing import Any


def record_skill(record: dict[str, Any]) -> Any:
    discovery = record.get("skill_discovery")
    if not isinstance(discovery, dict):
        discovery = {}
    return record.get("skill") or record.get("skill_name") or discovery.get("selected_skill")


def record_case(record: dict[str, Any]) -> Any:
    discovery = record.get("skill_discovery")
    if not isinstance(discovery, dict):
        discovery = {}
    return record.get("case_id") or discovery.get("selected_case_id")
