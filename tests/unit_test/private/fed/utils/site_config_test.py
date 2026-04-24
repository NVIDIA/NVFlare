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

from nvflare.private.fed.utils.site_config import load_site_config_from_file, validate_site_config


def test_validate_site_config_accepts_supported_shape():
    config, error = validate_site_config(
        {
            "resources": {"gpus": ["A100"], "memory_gb": 128},
            "labels": {"region": "us-east"},
            "capabilities": ["he", "psi"],
        }
    )

    assert error is None
    assert config["format_version"] == 1
    assert config["resources"]["memory_gb"] == 128


def test_validate_site_config_rejects_unsupported_top_level_key():
    config, error = validate_site_config({"format_version": 1, "private_path": "/tmp/secret"})

    assert config is None
    assert "unsupported top-level keys" in error


def test_validate_site_config_rejects_bad_capabilities():
    config, error = validate_site_config({"format_version": 1, "capabilities": ["he", 123]})

    assert config is None
    assert "capabilities must contain only strings" in error


def test_validate_site_config_rejects_bad_format_version_type():
    config, error = validate_site_config({"format_version": True, "labels": {"region": "us-east"}})

    assert config is None
    assert "format_version must be int" in error


def test_load_site_config_from_file_soft_fails_missing_file(tmp_path):
    config, error = load_site_config_from_file(str(tmp_path / "missing.json"))

    assert config is None
    assert error is None


def test_load_site_config_from_file_rejects_oversized_file(tmp_path):
    config_file = tmp_path / "site_config.json"
    config_file.write_text(json.dumps({"format_version": 1, "resources": {"data": "x" * 100}}))

    config, error = load_site_config_from_file(str(config_file), max_size=20)

    assert config is None
    assert "exceeds max size" in error
