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

"""Shared visual-QC vocabulary used by extraction and agent decisions."""

from typing import Any, Mapping

VISUAL_QC_TRANSFORMS = ("as_is", "hflip", "vflip", "rot180")
VISUAL_QC_TRANSFORM_SET = frozenset(VISUAL_QC_TRANSFORMS)


def visual_qc_decision_passed(
    decision: Mapping[str, Any] | None,
    *,
    label_orientation: Mapping[str, Any] | None = None,
) -> bool:
    """Return True only for a logically consistent, task-aligned QC pass."""

    if not isinstance(decision, Mapping):
        return False
    if decision.get("status") != "passed" or decision.get("passed") is not True:
        return False
    if decision.get("reviewed") is False:
        return False
    if decision.get("consensus_reached") is False:
        return False
    selected_transform = decision.get("selected_transform")
    if not isinstance(selected_transform, str) or selected_transform not in VISUAL_QC_TRANSFORM_SET:
        return False
    extracted_transform = "as_is"
    if isinstance(label_orientation, Mapping) and isinstance(label_orientation.get("selected_transform"), str):
        extracted_transform = str(label_orientation["selected_transform"])
    return selected_transform == extracted_transform


def visual_qc_result_ready_for_training(result: Mapping[str, Any]) -> bool:
    visual_qc = result.get("visual_qc") if isinstance(result, Mapping) else None
    label_orientation = result.get("label_orientation") if isinstance(result, Mapping) else None
    return visual_qc_decision_passed(
        visual_qc if isinstance(visual_qc, Mapping) else None,
        label_orientation=label_orientation if isinstance(label_orientation, Mapping) else None,
    )
