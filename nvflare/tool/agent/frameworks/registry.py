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

"""Framework detector registry.

Adding a framework with active detection = implement a ``FrameworkDetector`` in
its own module and append it to ``_DETECTORS``. Frameworks we only recognize by
import (no active class/call detection yet) live in ``IMPORT_ONLY_ROOTS`` until
a full detector and conversion skill land.
"""

from typing import Optional

from .base import FrameworkDetector
from .lightning import LightningDetector
from .pytorch import PyTorchDetector

# Detectors with active (class/call) detection, in dispatch order.
_DETECTORS: list[FrameworkDetector] = [
    PyTorchDetector(),
    LightningDetector(),
]

# Frameworks recognized by import only (ranked from import evidence) until a
# full detector + conversion skill is implemented. Keep the top-level import
# module name mapped to its framework bucket.
IMPORT_ONLY_ROOTS: dict[str, str] = {
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "xgboost": "xgboost",
    "sklearn": "sklearn",
    "jax": "jax",
    "flax": "jax",
    "optax": "jax",
    "numpy": "numpy",
}

# Aggregated top-level-module -> framework map (detectors + import-only).
_IMPORT_ROOTS: dict[str, str] = dict(IMPORT_ONLY_ROOTS)
for _detector in _DETECTORS:
    _IMPORT_ROOTS.update(_detector.import_roots)

# Aggregated evidence-kind -> ranking weight. "import" is the shared baseline.
_EVIDENCE_WEIGHTS: dict[str, int] = {"import": 1}
for _detector in _DETECTORS:
    _EVIDENCE_WEIGHTS.update(_detector.evidence_weights)


def detectors() -> list[FrameworkDetector]:
    return _DETECTORS


def evidence_weights() -> dict[str, int]:
    return _EVIDENCE_WEIGHTS


def framework_for_import(module: str) -> Optional[str]:
    """Map an imported module to its framework bucket by top-level segment."""
    if not module:
        return None
    return _IMPORT_ROOTS.get(module.split(".")[0])


def recommended_skill_for(framework: Optional[str]) -> Optional[str]:
    if framework is None:
        return None
    for detector in _DETECTORS:
        if detector.name == framework:
            return detector.recommended_skill
    return None


def _family_member_detectors() -> list[FrameworkDetector]:
    return [detector for detector in _DETECTORS if detector.family]


def _detector_by_name(name: str) -> Optional[FrameworkDetector]:
    for detector in _DETECTORS:
        if detector.name == name:
            return detector
    return None


def is_active_evidence(framework: str, evidence: dict) -> bool:
    detector = _detector_by_name(framework)
    if detector is None:
        return evidence.get("kind") != "import"
    return detector.is_active_evidence(evidence)


def resolve_primary_framework(primary: str, evidence_by_framework: dict, resolver) -> str:
    """Disambiguate a family conflict (e.g. PyTorch vs PyTorch Lightning).

    Returns the framework that should be primary. Only overrides ``primary``
    when it is part of a family whose base and member both have evidence; the
    member detector owns the promotion decision.
    """
    for member in _family_member_detectors():
        base = member.family
        if base in evidence_by_framework and member.name in evidence_by_framework:
            if primary in {base, member.name}:
                return member.name if member.promote_over_family(base, resolver) else base
    return primary


def family_member_of_base(base: Optional[str]) -> Optional[str]:
    """Return the family member whose base family is ``base`` (e.g. base ``pytorch`` -> ``pytorch_lightning``).

    Assumes at most one member per base (the current PyTorch family). If a base
    ever gains a second superset member, ordering and mixed-workspace detection
    would need to handle multiple members instead of the first match.
    """
    if base is None:
        return None
    for member in _family_member_detectors():
        if member.family == base:
            return member.name
    return None


def family_base_has_member(base: Optional[str], evidence_by_framework: dict) -> Optional[str]:
    """If ``base`` is a family base with a member present in evidence, return the member name."""
    if base is None:
        return None
    for member in _family_member_detectors():
        if member.family == base and base in evidence_by_framework and member.name in evidence_by_framework:
            return member.name
    return None
