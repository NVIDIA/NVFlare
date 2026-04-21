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

from nvflare.utils.job_launcher_utils import get_job_launcher_spec


class TestGetJobLauncherSpec:
    def test_site_mode_match(self):
        meta = {"launcher_spec": {"site-1": {"docker": {"image": "repo/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "repo/img:v1"}

    def test_site_not_present_returns_empty(self):
        meta = {"launcher_spec": {"site-1": {"docker": {"image": "repo/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-2", "docker") == {}

    def test_mode_not_present_returns_empty(self):
        meta = {"launcher_spec": {"site-1": {"k8s": {"image": "repo/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-1", "docker") == {}

    def test_default_used_when_no_site_entry(self):
        meta = {"launcher_spec": {"default": {"docker": {"image": "default/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "default/img:v1"}

    def test_site_overrides_default(self):
        meta = {
            "launcher_spec": {
                "default": {"docker": {"image": "default/img:v1", "shm_size": "4g"}},
                "site-1": {"docker": {"image": "site/img:v1"}},
            }
        }
        result = get_job_launcher_spec(meta, "site-1", "docker")
        assert result == {"image": "site/img:v1", "shm_size": "4g"}

    def test_default_only_applies_to_matching_mode(self):
        meta = {"launcher_spec": {"default": {"k8s": {"image": "repo/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-1", "docker") == {}

    def test_fallback_to_nested_resource_spec(self):
        meta = {"resource_spec": {"site-1": {"docker": {"image": "nested/img:v1"}}}}
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "nested/img:v1"}

    def test_no_launcher_spec_no_resource_spec_returns_empty(self):
        assert get_job_launcher_spec({}, "site-1", "docker") == {}

    def test_launcher_spec_takes_precedence_over_resource_spec(self):
        meta = {
            "launcher_spec": {"site-1": {"docker": {"image": "new/img:v1"}}},
            "resource_spec": {"site-1": {"docker": {"image": "old/img:v1"}}},
        }
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "new/img:v1"}
