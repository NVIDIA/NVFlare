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

"""Per-framework static-detection plugins for the agent inspector.

The inspector engine (``inspector.py``) stays framework-agnostic: it walks each
Python AST once and dispatches every node to the registered
``FrameworkDetector`` plugins. All framework-specific knowledge -- import
roots, symbols, evidence kinds and weights, the recommended conversion skill,
and cross-framework family/promotion rules -- lives in a detector module here.

To add a framework (XGBoost, TensorFlow, ...), add a detector module and
register it in ``registry.py``; the engine does not change.
"""

from .base import DetectContext, FrameworkDetector
from .registry import (
    IMPORT_ONLY_ROOTS,
    UTILITY_FRAMEWORKS,
    detectors,
    evidence_weights,
    fallback_skill_for,
    family_base_has_member,
    framework_for_import,
    has_active_family_member_conflict,
    is_active_evidence,
    is_training_owner_evidence,
    recommended_skill_for,
    resolve_primary_framework,
)

__all__ = [
    "DetectContext",
    "FrameworkDetector",
    "IMPORT_ONLY_ROOTS",
    "UTILITY_FRAMEWORKS",
    "detectors",
    "evidence_weights",
    "fallback_skill_for",
    "family_base_has_member",
    "framework_for_import",
    "has_active_family_member_conflict",
    "is_active_evidence",
    "is_training_owner_evidence",
    "recommended_skill_for",
    "resolve_primary_framework",
]
