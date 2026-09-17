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

from tests.integration_test.src.validators.json_stats_result_validator import JsonStatsResultValidator


def test_json_stats_validator_finds_nested_expected_relative_path(tmp_path):
    stats_file = tmp_path / "workspace" / "app" / "statistics" / "image_statistics.json"
    stats_file.parent.mkdir(parents=True)
    stats_file.write_text(json.dumps({"intensity": {"count": 1}}), encoding="utf-8")
    validator = JsonStatsResultValidator(
        relative_path="statistics/image_statistics.json", required_paths=["intensity.count"]
    )

    assert validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])


def test_json_stats_validator_rejects_partial_path_suffix_match(tmp_path):
    stats_file = tmp_path / "workspace" / "more_statistics" / "image_statistics.json"
    stats_file.parent.mkdir(parents=True)
    stats_file.write_text(json.dumps({"intensity": {"count": 1}}), encoding="utf-8")
    validator = JsonStatsResultValidator(
        relative_path="statistics/image_statistics.json", required_paths=["intensity.count"]
    )

    assert not validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])


def test_json_stats_validator_rejects_path_escape(tmp_path):
    outside_file = tmp_path / "outside.json"
    outside_file.write_text(json.dumps({"intensity": {"count": 1}}), encoding="utf-8")
    validator = JsonStatsResultValidator(relative_path="../outside.json", required_paths=["intensity.count"])

    assert not validator.validate_finished_results({"workspace_root": str(tmp_path / "workspace")}, [])
