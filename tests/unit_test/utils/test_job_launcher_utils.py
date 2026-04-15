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

from nvflare.utils.job_launcher_utils import extract_job_image


class TestExtractJobImage:
    def test_explicit_site_match(self):
        meta = {
            "deploy_map": {
                "app": [{"sites": ["client-1"], "image": "repo/img:v1"}]
            }
        }
        assert extract_job_image(meta, "client-1") == "repo/img:v1"

    def test_explicit_site_no_match(self):
        meta = {
            "deploy_map": {
                "app": [{"sites": ["client-1"], "image": "repo/img:v1"}]
            }
        }
        assert extract_job_image(meta, "client-2") is None

    def test_at_all_matches_any_site(self):
        meta = {
            "deploy_map": {
                "app": [{"sites": ["@ALL"], "image": "repo/img:v1"}]
            }
        }
        assert extract_job_image(meta, "client-1") == "repo/img:v1"
        assert extract_job_image(meta, "server") == "repo/img:v1"

    def test_string_item_returns_none(self):
        meta = {"deploy_map": {"app": ["@ALL"]}}
        assert extract_job_image(meta, "client-1") is None

    def test_empty_deploy_map(self):
        assert extract_job_image({}, "client-1") is None

    def test_multiple_entries_picks_correct_site(self):
        meta = {
            "deploy_map": {
                "app": [
                    {"sites": ["server"], "image": "gcr/img:v1"},
                    {"sites": ["client-1"], "image": "ecr/img:v1"},
                ]
            }
        }
        assert extract_job_image(meta, "server") == "gcr/img:v1"
        assert extract_job_image(meta, "client-1") == "ecr/img:v1"

    def test_at_all_with_explicit_override(self):
        meta = {
            "deploy_map": {
                "app": [
                    {"sites": ["client-1"], "image": "ecr/img:v1"},
                    {"sites": ["@ALL"], "image": "default/img:v1"},
                ]
            }
        }
        # explicit match takes precedence (appears first)
        assert extract_job_image(meta, "client-1") == "ecr/img:v1"
        # @ALL catches the rest
        assert extract_job_image(meta, "client-2") == "default/img:v1"
