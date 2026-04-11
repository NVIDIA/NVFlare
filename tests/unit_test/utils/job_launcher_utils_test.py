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
    def test_returns_image_for_matching_site(self):
        meta = {"deploy_map": {"app": [{"sites": ["server", "site-1"], "image": "nvflare:latest"}]}}
        assert extract_job_image(meta, "site-1") == "nvflare:latest"

    def test_returns_none_for_non_matching_site(self):
        meta = {"deploy_map": {"app": [{"sites": ["site-1"], "image": "nvflare:latest"}]}}
        assert extract_job_image(meta, "site-2") is None

    def test_at_all_matches_any_site(self):
        meta = {"deploy_map": {"app": [{"sites": ["@ALL"], "image": "nvflare:latest"}]}}
        assert extract_job_image(meta, "site-1") == "nvflare:latest"
        assert extract_job_image(meta, "server") == "nvflare:latest"

    def test_missing_sites_key_does_not_raise(self):
        """deploy_map entry with no 'sites' key must not raise TypeError."""
        meta = {"deploy_map": {"app": [{"image": "nvflare:latest"}]}}
        assert extract_job_image(meta, "site-1") is None

    def test_empty_deploy_map_returns_none(self):
        assert extract_job_image({"deploy_map": {}}, "site-1") is None

    def test_no_deploy_map_returns_none(self):
        assert extract_job_image({}, "site-1") is None

    def test_string_entry_in_deploy_map_is_skipped(self):
        """Plain string entries (e.g. '@ALL') in deploy_map are skipped — no image."""
        meta = {"deploy_map": {"app": ["@ALL"]}}
        assert extract_job_image(meta, "site-1") is None

    def test_per_site_image_takes_precedence_over_all(self):
        """Per-site entry matched first; @ALL entry is a fallback."""
        meta = {
            "deploy_map": {
                "app": [
                    {"sites": ["site-1"], "image": "site-specific:latest"},
                    {"sites": ["@ALL"], "image": "generic:latest"},
                ]
            }
        }
        assert extract_job_image(meta, "site-1") == "site-specific:latest"
