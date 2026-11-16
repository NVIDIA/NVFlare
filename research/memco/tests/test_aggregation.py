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

import unittest

from memco.core import aggregate_local_payloads


def _local_payload(agent_id: str, scene: str) -> dict:
    return {
        "agent_id": agent_id,
        "episode_ids": [f"episode-{agent_id}"],
        "candidates": [
            {
                "candidate_id": "workflow:search-then-finish",
                "candidate_type": "workflow_chunk",
                "summary": "Search for evidence, then finish with the supported label.",
                "structure": {
                    "pattern_kind": "workflow",
                    "task_family": "verification",
                    "workflow": ["search", "finish"],
                },
                "source_episode_ids": [f"episode-{agent_id}"],
                "source_scenes": [scene],
                "positive": 2,
                "negative": 0,
                "stalled": 0,
                "utility": 1.0,
                "wilson_episode_evidence": {
                    f"{scene}\x1fepisode-{agent_id}": {
                        "supporting": 2,
                        "contradicting": 0,
                        "stalled": 0,
                    }
                },
            }
        ],
        "rules": [],
        "artifacts": [],
    }


class AggregationTest(unittest.TestCase):
    def test_promotes_candidate_supported_by_multiple_clients(self):
        result = aggregate_local_payloads(
            [_local_payload("pddl", "pddl-train"), _local_payload("fever", "fever-train")]
        )

        self.assertEqual(1, result["candidate_count"])
        self.assertEqual(4, result["candidates"][0]["positive"])
        self.assertEqual(["fever-train", "pddl-train"], result["candidates"][0]["source_scenes"])
        self.assertEqual("wilson", result["promotion_policy"])
        self.assertEqual("wilson", result["promotion_report"]["active_policy"])

    def test_does_not_promote_single_client_candidate(self):
        result = aggregate_local_payloads([_local_payload("alfworld", "alfworld-train")])

        self.assertEqual(0, result["candidate_count"])


if __name__ == "__main__":
    unittest.main()
