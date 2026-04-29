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

import importlib.util
import logging
import sys

from nvflare.utils.job_launcher_utils import (
    _validate_launcher_spec,
    get_job_launcher_spec,
    refresh_custom_dir_import_path,
)


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

    def test_fallback_fires_when_launcher_spec_has_other_sites(self):
        # Fallback to resource_spec fires per (site, mode) pair, not only when
        # launcher_spec is entirely absent.
        meta = {
            "launcher_spec": {"site-2": {"docker": {"image": "site2/img:v1"}}},
            "resource_spec": {"site-1": {"docker": {"image": "legacy/img:v1"}}},
        }
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "legacy/img:v1"}

    def test_fallback_fires_when_launcher_spec_has_other_modes(self):
        meta = {
            "launcher_spec": {"site-1": {"k8s": {"image": "k8s/img:v1"}}},
            "resource_spec": {"site-1": {"docker": {"image": "legacy/img:v1"}}},
        }
        assert get_job_launcher_spec(meta, "site-1", "docker") == {"image": "legacy/img:v1"}


class TestValidateLauncherSpec:
    def test_clean_spec_returns_empty(self):
        spec = {
            "default": {"docker": {"image": "repo/img:v1"}},
            "site-1": {"docker": {"image": "repo/img:v1"}},
        }
        assert _validate_launcher_spec(spec) == []

    def test_typo_defaults_flagged(self):
        spec = {"defaults": {"docker": {"image": "repo/img:v1"}}}
        assert "defaults" in _validate_launcher_spec(spec)

    def test_typo_defaul_flagged(self):
        spec = {"defaul": {"docker": {"image": "repo/img:v1"}}}
        assert "defaul" in _validate_launcher_spec(spec)

    def test_exact_reserved_key_not_flagged(self):
        spec = {"default": {"docker": {"image": "repo/img:v1"}}}
        assert _validate_launcher_spec(spec) == []

    def test_unrelated_site_name_not_flagged(self):
        spec = {"site-1": {"docker": {"image": "repo/img:v1"}}}
        assert _validate_launcher_spec(spec) == []

    def test_non_dict_value_skipped(self):
        spec = {"defaults": "not-a-dict"}
        assert _validate_launcher_spec(spec) == []

    def test_warning_emitted_for_suspicious_key(self, caplog):
        meta = {"launcher_spec": {"defaults": {"docker": {"image": "repo/img:v1"}}}}
        with caplog.at_level(logging.WARNING, logger="nvflare.utils.job_launcher_utils"):
            get_job_launcher_spec(meta, "site-1", "docker")
        assert any("defaults" in msg for msg in caplog.messages)


class TestRefreshCustomDirImportPath:
    def test_logs_when_custom_dir_is_missing(self, tmp_path, caplog):
        custom_path = str(tmp_path / "missing" / "custom")

        with caplog.at_level(logging.DEBUG, logger="nvflare.utils.job_launcher_utils"):
            refresh_custom_dir_import_path(custom_path)

        assert "custom dir not found" in caplog.text
        assert custom_path in caplog.text

    def test_refreshes_importer_cache_for_dir_created_after_startup(self, tmp_path):
        module_name = "nvflare_refresh_path_probe"
        custom_dir = tmp_path / "app" / "custom"
        custom_path = str(custom_dir)
        sys.path.append(custom_path)
        try:
            assert importlib.util.find_spec(module_name) is None
            assert sys.path_importer_cache.get(custom_path) is None

            custom_dir.mkdir(parents=True)
            (custom_dir / f"{module_name}.py").write_text("VALUE = 123\n")

            assert importlib.util.find_spec(module_name) is None
            refresh_custom_dir_import_path(custom_path)

            spec = importlib.util.find_spec(module_name)
            assert spec is not None
            assert spec.origin == str(custom_dir / f"{module_name}.py")
        finally:
            sys.modules.pop(module_name, None)
            if custom_path in sys.path:
                sys.path.remove(custom_path)
            sys.path_importer_cache.pop(custom_path, None)
