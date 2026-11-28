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

import json
import tempfile
import unittest
from pathlib import Path

from memco.domains import load_subset_file, normalize_pddl_rows, split_fever_by_domain, summarize_task_results


class DomainDataTest(unittest.TestCase):
    def test_partial_goal_reward_does_not_count_as_task_success(self):
        result = summarize_task_results([{"reward": 0.5, "done": False}, {"reward": 1.0, "done": True}])
        self.assertEqual(1, result["num_success"])
        self.assertEqual(0.5, result["accuracy"])
        self.assertEqual(0.75, result["avg_reward"])

    def test_empty_results_have_zero_metrics(self):
        result = summarize_task_results([])
        self.assertEqual(0, result["num_completed"])
        self.assertEqual(0.0, result["accuracy"])
        self.assertEqual(0.0, result["avg_reward"])

    def test_pddl_indexing_matches_file_order_per_game(self):
        rows = [
            {"additional_info": {"subtask": "Gripper"}, "difficulty": None},
            {"game_name": "gripper"},
            {"game_name": "barman", "problem_index": "4"},
            {"game_name": "barman"},
        ]

        result = normalize_pddl_rows(rows)

        self.assertEqual([0, 1, 4, 0], [row["problem_index"] for row in result])
        self.assertEqual(["gripper", "gripper", "barman", "barman"], [row["game_name"] for row in result])
        self.assertEqual("", result[0]["difficulty"])

    def test_fever_split_preserves_input_order(self):
        rows = [
            {"claim": "A", "label": "SUPPORTS", "ab_domain": "A_film_tv"},
            {"claim": "B", "label": "REFUTES", "ab_domain": "B_music"},
            {"claim": "C", "label": "SUPPORTS", "ab_domain": "A_film_tv"},
        ]

        result = split_fever_by_domain(rows, ["A_film_tv", "B_music"])

        self.assertEqual(["A", "C"], [row["task"] for row in result["A_film_tv"]])
        self.assertEqual(["B"], [row["task"] for row in result["B_music"]])

    def test_alfworld_subset_uses_exact_domain_and_split_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            subset_dir = Path(directory)
            expected = [{"scene_domain": "bathroom", "source_split": "valid_seen"}]
            (subset_dir / "bathroom__valid_seen.json").write_text(json.dumps(expected), encoding="utf-8")

            result = load_subset_file(subset_dir, "bathroom", "valid_seen")

        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
