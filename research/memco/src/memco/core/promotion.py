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

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

from .graph_types import ArtifactKind, CandidateType, MemoryArtifact, MemoryRule, PromotionCandidate, RuleType

PROMOTION_POLICIES: Final[tuple[str, ...]] = ("legacy", "shadow", "wilson")


def normalize_promotion_policy(value: str) -> str:
    policy = str(value or "legacy").strip().lower()
    if policy not in PROMOTION_POLICIES:
        choices = ", ".join(PROMOTION_POLICIES)
        raise ValueError(f"unknown promotion policy {value!r}; expected one of: {choices}")
    return policy


def wilson_lower_bound(
    supporting: int,
    contradicting: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the one-sided Wilson lower bound for a supporting relation.

    ``supporting`` and ``contradicting`` are the two mutually exclusive
    relation verdicts in Eq. (11).  A stalled flag is descriptive metadata on
    a verdict and therefore is not added to the binomial denominator.
    """

    if supporting < 0 or contradicting < 0:
        raise ValueError("Wilson evidence counts must be non-negative")
    if not 0.0 < alpha < 0.5:
        raise ValueError("Wilson alpha must lie strictly between 0 and 0.5")
    sample_size = supporting + contradicting
    if sample_size == 0:
        return 0.0

    z_value = NormalDist().inv_cdf(1.0 - alpha)
    observed_rate = supporting / sample_size
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / sample_size
    center = observed_rate + z_squared / (2.0 * sample_size)
    margin = (
        z_value
        * (observed_rate * (1.0 - observed_rate) / sample_size + z_squared / (4.0 * sample_size * sample_size)) ** 0.5
    )
    return max(0.0, (center - margin) / denominator)


@dataclass(frozen=True, slots=True)
class WilsonEvidence:
    supporting: int
    contradicting: int
    stalled: int
    source_coverage: int
    evidence_unit: str = "episode"

    def __post_init__(self) -> None:
        if min(self.supporting, self.contradicting, self.stalled, self.source_coverage) < 0:
            raise ValueError("evidence statistics must be non-negative")
        if self.stalled > self.sample_size:
            raise ValueError("stalled evidence must annotate a matched evidence trial")

    @property
    def sample_size(self) -> int:
        return self.supporting + self.contradicting

    @property
    def confidence(self) -> float:
        return self.supporting / self.sample_size if self.sample_size else 0.0

    @property
    def negative_rate(self) -> float:
        return self.contradicting / self.sample_size if self.sample_size else 0.0

    @property
    def stalled_rate(self) -> float:
        return self.stalled / self.sample_size if self.sample_size else 0.0

    def score(self, *, alpha: float) -> float:
        return wilson_lower_bound(self.supporting, self.contradicting, alpha=alpha)

    def to_dict(self, *, alpha: float) -> dict[str, float | int | str]:
        return {
            "n_sup": self.supporting,
            "n_con": self.contradicting,
            "n_stall": self.stalled,
            "s": self.sample_size,
            "conf": self.confidence,
            "neg": self.negative_rate,
            "rho": self.stalled_rate,
            "d": self.source_coverage,
            "evidence_unit": self.evidence_unit,
            "wilson_lower_bound": self.score(alpha=alpha),
        }


def _episode_level_counts(
    episode_evidence: dict[str, dict[str, int]],
    *,
    raw_supporting: int,
    raw_contradicting: int,
    raw_stalled: int,
    stalled_supports: bool = False,
) -> tuple[int, int, int, str]:
    """Collapse repeated record matches to one relation verdict per episode.

    A contradicting observation takes precedence when an ordinary relation has
    mixed outcomes in one episode.  For an anti-pattern, a stalled observation
    supports the relation and takes precedence.  Residual aggregate counts keep
    old snapshots (which predate the episode ledger) usable without changing
    the legacy counters.
    """

    if not episode_evidence:
        if stalled_supports:
            supporting = min(raw_stalled, raw_contradicting)
            contradicting = max(raw_contradicting - supporting, 0) + raw_supporting
        else:
            supporting = raw_supporting
            contradicting = raw_contradicting
        sample_size = supporting + contradicting
        return supporting, contradicting, min(raw_stalled, sample_size), "occurrence_fallback"

    supporting = 0
    contradicting = 0
    stalled = 0
    ledger_supporting = 0
    ledger_contradicting = 0
    ledger_stalled = 0
    for verdict in episode_evidence.values():
        observed_supporting = max(int((verdict or {}).get("supporting", 0)), 0)
        observed_contradicting = max(int((verdict or {}).get("contradicting", 0)), 0)
        observed_stalled = max(int((verdict or {}).get("stalled", 0)), 0)
        ledger_supporting += observed_supporting
        ledger_contradicting += observed_contradicting
        ledger_stalled += observed_stalled
        if observed_stalled:
            stalled += 1
        if stalled_supports:
            if observed_stalled:
                supporting += 1
            elif observed_supporting or observed_contradicting:
                contradicting += 1
        elif observed_contradicting:
            contradicting += 1
        elif observed_supporting:
            supporting += 1

    # A memory loaded from an older snapshot can receive new observations.  In
    # that mixed case, preserve only the unmatched historical aggregate counts
    # as an occurrence-level fallback; new observations still use episodes.
    residual_supporting = max(raw_supporting - ledger_supporting, 0)
    residual_contradicting = max(raw_contradicting - ledger_contradicting, 0)
    residual_stalled = max(raw_stalled - ledger_stalled, 0)
    has_residual = bool(residual_supporting or residual_contradicting or residual_stalled)
    if stalled_supports:
        residual_anti_support = min(residual_stalled, residual_contradicting)
        supporting += residual_anti_support
        contradicting += max(residual_contradicting - residual_anti_support, 0) + residual_supporting
    else:
        supporting += residual_supporting
        contradicting += residual_contradicting
    stalled += residual_stalled
    sample_size = supporting + contradicting
    return (
        supporting,
        contradicting,
        min(stalled, sample_size),
        "episode_with_occurrence_fallback" if has_residual else "episode",
    )


def candidate_wilson_evidence(candidate: PromotionCandidate) -> WilsonEvidence:
    pattern_kind = str(candidate.structure.get("pattern_kind", "") or "")
    anti_pattern = candidate.candidate_type == CandidateType.FAILURE and pattern_kind == "anti_pattern"
    supporting, contradicting, stalled, evidence_unit = _episode_level_counts(
        candidate.wilson_episode_evidence,
        raw_supporting=candidate.positive,
        raw_contradicting=candidate.negative,
        raw_stalled=candidate.stalled,
        stalled_supports=anti_pattern,
    )
    return WilsonEvidence(
        supporting=supporting,
        contradicting=contradicting,
        stalled=stalled,
        source_coverage=candidate.coverage,
        evidence_unit=evidence_unit,
    )


def rule_wilson_evidence(rule: MemoryRule) -> WilsonEvidence:
    supporting, contradicting, stalled, evidence_unit = _episode_level_counts(
        rule.wilson_episode_evidence,
        raw_supporting=rule.stats.success,
        raw_contradicting=rule.stats.failure,
        raw_stalled=rule.stats.stalled,
    )
    return WilsonEvidence(
        supporting=supporting,
        contradicting=contradicting,
        stalled=stalled,
        source_coverage=rule.coverage,
        evidence_unit=evidence_unit,
    )


def artifact_wilson_evidence(artifact: MemoryArtifact) -> WilsonEvidence:
    supporting, contradicting, stalled, evidence_unit = _episode_level_counts(
        artifact.wilson_episode_evidence,
        raw_supporting=artifact.stats.success,
        raw_contradicting=artifact.stats.failure,
        raw_stalled=artifact.stats.stalled,
    )
    return WilsonEvidence(
        supporting=supporting,
        contradicting=contradicting,
        stalled=stalled,
        source_coverage=artifact.coverage,
        evidence_unit=evidence_unit,
    )


def candidate_is_structurally_valid(candidate: PromotionCandidate) -> bool:
    structure = candidate.structure
    pattern_kind = str(structure.get("pattern_kind", "") or "")
    task_family = str(structure.get("task_family", "") or "").strip()
    if not task_family:
        return False
    if candidate.candidate_type == CandidateType.PRECONDITION:
        return bool(structure.get("action")) and bool(structure.get("precondition") or structure.get("from"))
    if candidate.candidate_type == CandidateType.WORKFLOW:
        workflow = tuple(structure.get("workflow", ()) or ())
        return pattern_kind in {"workflow", "closure"} and len(workflow) >= 2
    if candidate.candidate_type == CandidateType.REPAIR:
        return bool(structure.get("failure_label") and structure.get("repair_action"))
    if candidate.candidate_type == CandidateType.FAILURE:
        if pattern_kind == "anti_pattern":
            return bool(tuple(structure.get("anti_pattern", ()) or ()))
        # An explicit failure is evidence used to construct blocked/repair
        # knowledge, not itself a transferable top-level relation.
        return False
    return False


def rule_is_structurally_valid(rule: MemoryRule) -> bool:
    if rule.rule_type in {RuleType.BLOCKED, RuleType.REPAIR}:
        return False
    return bool(rule.task_family and rule.condition and rule.effect)


def artifact_is_structurally_valid(artifact: MemoryArtifact) -> bool:
    if artifact.kind == ArtifactKind.REFLECTION:
        return False
    return bool(artifact.anchor and artifact.payload)
