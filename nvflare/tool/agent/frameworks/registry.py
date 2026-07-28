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
from .huggingface import HuggingFaceDetector
from .lightning import LightningDetector
from .pytorch import PyTorchDetector

# Detectors with active (class/call) detection, in dispatch order.
_DETECTORS: list[FrameworkDetector] = [
    PyTorchDetector(),
    LightningDetector(),
    HuggingFaceDetector(),
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

# Numerical/array utilities that are used by virtually every ML framework rather
# than being the primary training framework. Their mere presence (typically an
# incidental import) must not win primary-framework selection over a real
# convertible framework whose code is loaded dynamically or lives outside the
# entry file. These are still ranked/reported, just never promoted as the
# entry-context primary.
UTILITY_FRAMEWORKS: frozenset[str] = frozenset({"numpy"})

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


def recommended_skill_for(framework: Optional[str], evidence: Optional[list[dict]] = None) -> Optional[str]:
    evidence = evidence or []
    if framework is None:
        return None
    for detector in _DETECTORS:
        if detector.name == framework:
            if detector.recommendation_requires_active_evidence and not any(
                detector.is_active_evidence(item) for item in evidence
            ):
                return None
            return detector.recommended_skill
    return None


def fallback_skill_for(framework: Optional[str], evidence: list[dict]) -> Optional[str]:
    detector = _detector_by_name(framework) if framework else None
    return detector.fallback_skill_for(evidence) if detector else None


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


def is_training_owner_evidence(framework: str, evidence: dict) -> bool:
    detector = _detector_by_name(framework)
    if detector is None:
        return False
    return detector.is_training_owner_evidence(evidence)


def is_candidate_evidence(framework: str, evidence: dict) -> bool:
    detector = _detector_by_name(framework)
    if detector is None:
        return False
    return detector.is_candidate_evidence(evidence)


def has_active_family_member_conflict(evidence_by_framework: dict, resolver) -> bool:
    """Whether reachable training owners from multiple specialized members claim one family."""
    active_by_family: dict[str, int] = {}
    for detector in _family_member_detectors():
        family = detector.family
        if not family:
            continue
        evidence = evidence_by_framework.get(detector.name, [])
        owner_evidence = [item for item in evidence if detector.is_training_owner_evidence(item)]
        if owner_evidence and resolver.tied_to_entry_context(owner_evidence):
            active_by_family[family] = active_by_family.get(family, 0) + 1
    return any(count > 1 for count in active_by_family.values())


def ownership_summary(
    detected_framework: Optional[str],
    evidence_by_framework: dict,
    resolver,
    *,
    family_member_conflict: bool,
) -> dict:
    """Build an additive, framework-neutral ownership summary."""
    evidence_owners = []
    candidates = []
    for framework in evidence_by_framework:
        owner_evidence = resolver.training_owner_evidence(framework)
        candidate_evidence = resolver.candidate_evidence(framework)
        if owner_evidence and resolver.tied_to_entry_context(owner_evidence):
            evidence_owners.append(framework)
        elif candidate_evidence and resolver.tied_to_entry_context(candidate_evidence):
            candidates.append(framework)

    if family_member_conflict:
        state = "conflicting"
        owners = evidence_owners
    elif detected_framework in evidence_owners:
        state = "clear"
        # Family resolution may absorb base-framework activity into one
        # specialized owner. The full evidence remains in ``frameworks``;
        # this field reports the resolved owner.
        owners = [detected_framework]
    elif evidence_owners:
        state = "unresolved"
        owners = evidence_owners
    elif candidates:
        state = "candidate"
        owners = []
    elif evidence_by_framework:
        state = "import_only"
        owners = []
    else:
        state = "none"
        owners = []
    return {"state": state, "owners": sorted(owners), "candidates": sorted(candidates)}


def resolve_primary_framework(primary: str, evidence_by_framework: dict, resolver) -> str:
    """Disambiguate a family conflict (e.g. PyTorch vs PyTorch Lightning).

    Returns the framework that should be primary. Only overrides ``primary``
    when it is part of a family whose base and member both have evidence; the
    member detector owns the promotion decision.
    """
    primary_detector = _detector_by_name(primary)
    family_base = primary_detector.family if primary_detector and primary_detector.family else primary
    owner_members = [
        member
        for member in _family_member_detectors()
        if member.family == family_base
        and (owner_evidence := resolver.training_owner_evidence(member.name))
        and resolver.tied_to_entry_context(owner_evidence)
    ]
    if len(owner_members) == 1:
        return owner_members[0].name
    if len(owner_members) > 1:
        owner_names = {member.name for member in owner_members}
        return primary if primary in owner_names else owner_members[0].name

    if primary_detector and primary_detector.family:
        base = family_base
        if base in evidence_by_framework:
            if primary_detector.promote_over_family(base, resolver):
                return primary
            promoted = _promoted_family_member(base, evidence_by_framework, resolver, exclude={primary})
            return promoted.name if promoted else base
        return primary

    promoted = _promoted_family_member(primary, evidence_by_framework, resolver)
    if promoted:
        return promoted.name
    return primary


def _promoted_family_member(
    base: str,
    evidence_by_framework: dict,
    resolver,
    *,
    exclude: Optional[set[str]] = None,
) -> Optional[FrameworkDetector]:
    exclude = exclude or set()
    candidates = [
        member
        for member in _family_member_detectors()
        if member.name not in exclude
        and member.family == base
        and member.name in evidence_by_framework
        and member.promote_over_family(base, resolver)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda member: resolver.score(resolver.evidence(member.name)))


def family_base_has_member(base: Optional[str], evidence_by_framework: dict) -> Optional[str]:
    """If ``base`` is a family base with a member present in evidence, return the member name."""
    if base is None:
        return None
    for member in _family_member_detectors():
        if member.family == base and base in evidence_by_framework and member.name in evidence_by_framework:
            return member.name
    return None
