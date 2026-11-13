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

import unittest

from memco.core.construction_graph import GlobalPromoter
from memco.core.graph_types import CandidateType, GlobalGraphMemory, LocalGraphMemory, PromotionCandidate
from memco.core.promotion import candidate_wilson_evidence, wilson_lower_bound
from memco.core.serialization import _candidate_to_dict, candidate_from_dict, global_memory_to_dict


def _precondition_candidate(*, positives: int, negatives: int = 0) -> PromotionCandidate:
    candidate = PromotionCandidate(
        candidate_id="precondition:test:open",
        candidate_type=CandidateType.PRECONDITION,
        summary="Being at a container enables opening it.",
        structure={
            "precondition": {"location_matches_container": True},
            "action": "open(container)",
            "task_family": "test_family",
            "pattern_kind": "precondition",
        },
    )
    for index in range(positives):
        candidate.observe(
            scene_id=f"scene_{index % 2}",
            episode_id=f"positive_{index}",
            positive=True,
        )
    for index in range(negatives):
        candidate.observe(
            scene_id=f"scene_{index % 2}",
            episode_id=f"negative_{index}",
            positive=False,
        )
    return candidate


class WilsonScoreTest(unittest.TestCase):
    def test_one_sided_lower_bound_known_values(self) -> None:
        self.assertAlmostEqual(wilson_lower_bound(0, 0), 0.0)
        self.assertAlmostEqual(wilson_lower_bound(2, 0), 0.4250306, places=6)
        self.assertAlmostEqual(wilson_lower_bound(3, 0), 0.5258044, places=6)
        self.assertAlmostEqual(wilson_lower_bound(3, 1), 0.3561680, places=6)

    def test_antipattern_stall_is_support_for_stall_relation(self) -> None:
        candidate = PromotionCandidate(
            candidate_id="anti:test",
            candidate_type=CandidateType.FAILURE,
            summary="Repeated search stalls.",
            structure={
                "anti_pattern": ("search", "search"),
                "task_family": "test_family",
                "pattern_kind": "anti_pattern",
            },
            negative=3,
            stalled=3,
            source_scenes={"scene_a", "scene_b"},
        )
        evidence = candidate_wilson_evidence(candidate)
        self.assertEqual(evidence.supporting, 3)
        self.assertEqual(evidence.contradicting, 0)
        self.assertEqual(evidence.stalled, 3)
        self.assertEqual(evidence.sample_size, 3)

    def test_repeated_matches_in_one_episode_form_one_trial(self) -> None:
        candidate = _precondition_candidate(positives=0)
        for _ in range(3):
            candidate.observe(
                scene_id="scene_a",
                episode_id="episode_a",
                positive=True,
            )
        evidence = candidate_wilson_evidence(candidate)
        self.assertEqual(evidence.supporting, 1)
        self.assertEqual(evidence.contradicting, 0)
        self.assertEqual(evidence.evidence_unit, "episode")

    def test_mixed_episode_is_one_contradicting_trial(self) -> None:
        candidate = _precondition_candidate(positives=0)
        candidate.observe(
            scene_id="scene_a",
            episode_id="episode_a",
            positive=True,
        )
        candidate.observe(
            scene_id="scene_a",
            episode_id="episode_a",
            positive=False,
        )
        evidence = candidate_wilson_evidence(candidate)
        self.assertEqual(evidence.supporting, 0)
        self.assertEqual(evidence.contradicting, 1)

    def test_old_snapshot_counts_remain_usable(self) -> None:
        candidate = _precondition_candidate(positives=0)
        candidate.positive = 3
        candidate.source_scenes = {"scene_a"}
        evidence = candidate_wilson_evidence(candidate)
        self.assertEqual(evidence.supporting, 3)
        self.assertEqual(evidence.evidence_unit, "occurrence_fallback")

    def test_episode_ledger_round_trip(self) -> None:
        candidate = _precondition_candidate(positives=3)
        restored = candidate_from_dict(_candidate_to_dict(candidate))
        self.assertEqual(
            restored.wilson_episode_evidence,
            candidate.wilson_episode_evidence,
        )


class PromotionPolicyTest(unittest.TestCase):
    def test_default_and_explicit_legacy_are_identical(self) -> None:
        local = LocalGraphMemory(agent_id="agent")
        candidate = _precondition_candidate(positives=3)
        local.candidates[candidate.candidate_id] = candidate

        default_result = GlobalPromoter().promote(GlobalGraphMemory(), [local], batch_name="test")
        explicit_result = GlobalPromoter(policy="legacy").promote(GlobalGraphMemory(), [local], batch_name="test")
        self.assertEqual(global_memory_to_dict(default_result), global_memory_to_dict(explicit_result))

    def test_shadow_returns_legacy_and_keeps_wilson_separate(self) -> None:
        local = LocalGraphMemory(agent_id="agent")
        candidate = _precondition_candidate(positives=3)
        local.candidates[candidate.candidate_id] = candidate

        expected_legacy = GlobalPromoter(policy="legacy").promote(GlobalGraphMemory(), [local], batch_name="test")
        promoter = GlobalPromoter(policy="shadow")
        active = promoter.promote(GlobalGraphMemory(), [local], batch_name="test")

        self.assertEqual(global_memory_to_dict(active), global_memory_to_dict(expected_legacy))
        self.assertIsNotNone(promoter.last_wilson_memory)
        assert promoter.last_wilson_memory is not None
        self.assertIn(candidate.candidate_id, promoter.last_wilson_memory.candidates)
        self.assertEqual(promoter.last_promotion_report["active_policy"], "legacy")
        decision = promoter.last_promotion_report["decisions"]["candidates"][0]
        self.assertEqual(decision["s"], 3)
        self.assertEqual(decision["n_sup"], 3)
        self.assertEqual(decision["n_con"], 0)
        self.assertEqual(decision["d"], 2)

    def test_wilson_threshold_accounts_for_sample_uncertainty(self) -> None:
        two_trial_local = LocalGraphMemory(agent_id="two")
        two_trial_candidate = _precondition_candidate(positives=2)
        two_trial_local.candidates[two_trial_candidate.candidate_id] = two_trial_candidate
        two_trial_result = GlobalPromoter(policy="wilson").promote(
            GlobalGraphMemory(), [two_trial_local], batch_name="two"
        )
        self.assertNotIn(two_trial_candidate.candidate_id, two_trial_result.candidates)

        three_trial_local = LocalGraphMemory(agent_id="three")
        three_trial_candidate = _precondition_candidate(positives=3)
        three_trial_local.candidates[three_trial_candidate.candidate_id] = three_trial_candidate
        three_trial_result = GlobalPromoter(policy="wilson").promote(
            GlobalGraphMemory(), [three_trial_local], batch_name="three"
        )
        self.assertIn(three_trial_candidate.candidate_id, three_trial_result.candidates)

    def test_source_coverage_is_diagnostic_not_a_gate(self) -> None:
        local = LocalGraphMemory(agent_id="one_scene")
        candidate = _precondition_candidate(positives=0)
        for index in range(3):
            candidate.observe(
                scene_id="scene_a",
                episode_id=f"episode_{index}",
                positive=True,
            )
        local.candidates[candidate.candidate_id] = candidate
        promoter = GlobalPromoter(policy="wilson", wilson_min_coverage=99)
        result = promoter.promote(GlobalGraphMemory(), [local], batch_name="test")
        self.assertIn(candidate.candidate_id, result.candidates)
        decision = promoter.last_promotion_report["decisions"]["candidates"][0]
        self.assertEqual(decision["d"], 1)
        self.assertNotIn("insufficient_source_coverage", decision["reasons"])

    def test_raw_failure_candidate_is_not_promoted(self) -> None:
        local = LocalGraphMemory(agent_id="agent")
        candidate = PromotionCandidate(
            candidate_id="failure:test",
            candidate_type=CandidateType.FAILURE,
            summary="Opening the object failed.",
            structure={
                "failure_label": "not_openable",
                "action": "open(object)",
                "task_family": "test_family",
                "pattern_kind": "failure",
            },
        )
        for index in range(3):
            candidate.observe(
                scene_id="scene_a",
                episode_id=f"episode_{index}",
                positive=True,
            )
        local.candidates[candidate.candidate_id] = candidate
        promoter = GlobalPromoter(policy="wilson")
        result = promoter.promote(GlobalGraphMemory(), [local], batch_name="test")
        self.assertNotIn(candidate.candidate_id, result.candidates)
        decision = promoter.last_promotion_report["decisions"]["candidates"][0]
        self.assertIn("invalid_structure", decision["reasons"])

    def test_wilson_rebuild_drops_stale_global_records(self) -> None:
        stale = _precondition_candidate(positives=3)
        existing = GlobalGraphMemory(candidates={stale.candidate_id: stale})
        result = GlobalPromoter(policy="wilson").promote(
            existing,
            [],
            batch_name="empty",
        )
        self.assertNotIn(stale.candidate_id, result.candidates)


if __name__ == "__main__":
    unittest.main()
