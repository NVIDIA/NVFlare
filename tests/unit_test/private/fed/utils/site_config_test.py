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

from nvflare.private.fed.utils.site_config import SITE_CONFIG_EXCLUDED_TOP_LEVEL_KEYS, project_site_config


def test_drops_blacklisted_top_level_keys():
    config_data = {
        "format_version": 2,
        "client": {"retry_timeout": 30},
        "components": [{"id": "x"}],
        "handlers": [{"id": "h"}],
        "servers": [{"name": "s"}],
        "snapshot_persistor": {"path": "/tmp/x"},
        "admin": {"port": 8003},
        "relay_config": {"fqcn": "relay.x"},
        "overseer_agent": {"path": "x.OverseerAgent", "args": {}},
    }
    assert project_site_config(config_data) == {}


def test_keeps_custom_top_level_keys():
    config_data = {
        "format_version": 2,
        "client": {"retry_timeout": 30},
        "components": [{"id": "x"}],
        "labels": {"region": "us-east"},
        "capabilities": ["he", "psi"],
        "resources": {"memory_gb": 128},
    }
    projected = project_site_config(config_data)
    assert projected == {
        "labels": {"region": "us-east"},
        "capabilities": ["he", "psi"],
        "resources": {"memory_gb": 128},
    }


def test_result_is_deep_copied():
    config_data = {"labels": {"region": "us-east"}}
    projected = project_site_config(config_data)

    config_data["labels"]["region"] = "mutated"
    projected["labels"]["region"] = "also-mutated"

    assert config_data["labels"]["region"] == "mutated"
    assert projected["labels"]["region"] == "also-mutated"


def test_returns_empty_for_non_dict():
    assert project_site_config(None) == {}
    assert project_site_config("not a dict") == {}


def test_blacklist_covers_known_local_keys():
    # Guard against accidental removal of an exclusion entry.
    for key in (
        "format_version",
        "client",
        "servers",
        "components",
        "handlers",
        "snapshot_persistor",
        "admin",
        "relay_config",
        "overseer_agent",
    ):
        assert key in SITE_CONFIG_EXCLUDED_TOP_LEVEL_KEYS
