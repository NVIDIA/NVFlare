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

"""Framework-independent aggregation for federated MemCo artifacts."""

from collections.abc import Iterable

from .construction_graph import GlobalPromoter
from .graph_types import GlobalGraphMemory
from .serialization import global_memory_to_dict, local_memory_from_dict


def aggregate_local_payloads(
    payloads: Iterable[dict],
    *,
    promotion_threshold: float = 0.35,
    promotion_policy: str = "wilson",
    wilson_alpha: float = 0.05,
    wilson_threshold: float = 0.35,
    batch_name: str = "federated_round",
) -> dict:
    """Promote a collection of client-local memory payloads into global memory."""
    local_memories = [local_memory_from_dict(payload) for payload in payloads]
    promoter = GlobalPromoter(
        score_threshold=float(promotion_threshold),
        policy=promotion_policy,
        wilson_alpha=float(wilson_alpha),
        wilson_threshold=float(wilson_threshold),
    )
    global_memory = promoter.promote(GlobalGraphMemory(), local_memories, batch_name=batch_name)
    result = global_memory_to_dict(global_memory)
    result["promotion_policy"] = promoter.policy
    result["promotion_report"] = promoter.last_promotion_report
    if promoter.last_wilson_memory is not None and promoter.policy == "shadow":
        result["wilson_shadow_memory"] = global_memory_to_dict(promoter.last_wilson_memory)
    return result
