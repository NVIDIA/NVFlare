# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.apis.job_def import (
    DEFAULT_STUDY,
    MAX_JOB_OUTPUTS,
    get_job_meta_study,
    is_valid_job_id,
    job_output_matches,
    validate_job_outputs,
)


class TestJobDef:
    def test_is_valid_job_id(self):
        assert not is_valid_job_id("site-1")
        assert is_valid_job_id("c2564481-536a-4548-8dfa-cf183a3652a1")
        assert is_valid_job_id("c2564481536a45488dfacf183a3652a1")
        assert not is_valid_job_id("c2564481536a45488dfacf183a3652a1ddd")
        assert not is_valid_job_id("c2564481-536a-4548-fdff-df183a3652a1")

    def test_get_job_meta_study_defaults_legacy_jobs(self):
        assert get_job_meta_study({}) == DEFAULT_STUDY


class TestJobOutputs:
    def test_accepts_files_and_folders_and_removes_duplicates(self):
        outputs = ["app_server/FL_global_model.pt", "metrics/", "app_server/FL_global_model.pt", "output"]
        assert validate_job_outputs(outputs) == ["app_server/FL_global_model.pt", "metrics/", "output"]

    @pytest.mark.parametrize(
        "entry",
        ["", "/abs/path", "C:/model.pt", "a\\b", "../escape", "a/../b", "./a", "a/./b", "a//b", "/", "a/b//"],
    )
    def test_rejects_unsafe_entries(self, entry):
        with pytest.raises(ValueError):
            validate_job_outputs([entry])

    @pytest.mark.parametrize("value", ["metrics/", {"a": 1}, None, ["ok", 3]])
    def test_rejects_non_string_lists(self, value):
        with pytest.raises(TypeError):
            validate_job_outputs(value)

    def test_rejects_oversized_lists(self):
        with pytest.raises(ValueError, match="at most"):
            validate_job_outputs([f"f{i}" for i in range(MAX_JOB_OUTPUTS + 1)])

    @pytest.mark.parametrize(
        "member, expected",
        [
            ("metrics/metrics_summary.json", "metrics/"),
            ("metrics2/x.json", ""),
            ("app_server/FL_global_model.pt", "app_server/FL_global_model.pt"),
            ("app_server/FL_global_model.pt.bak", ""),
        ],
    )
    def test_matches_exact_files_and_folder_prefixes(self, member, expected):
        assert job_output_matches(member, ["metrics/", "app_server/FL_global_model.pt"]) == expected
