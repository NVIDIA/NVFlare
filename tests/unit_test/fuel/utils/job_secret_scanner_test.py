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
import warnings

import pytest

from nvflare.fuel.utils.job_secret_scanner import warn_on_potential_secrets_in_job_dir
from nvflare.fuel.utils.secret_utils import PotentialSecretWarning


def test_warns_on_generated_config_without_exposing_value(tmp_path):
    config_dir = tmp_path / "job" / "app" / "config"
    config_dir.mkdir(parents=True)
    secret = "ghp_" + "Ab1" * 12
    (config_dir / "config_fed_client.json").write_text(json.dumps({"token": secret}))

    with pytest.warns(PotentialSecretWarning) as record:
        findings = warn_on_potential_secrets_in_job_dir(str(tmp_path), job_name="job")

    assert findings
    assert all(secret not in str(warning.message) for warning in record)


def test_ignores_custom_source_files(tmp_path):
    custom_dir = tmp_path / "job" / "app" / "custom"
    custom_dir.mkdir(parents=True)
    (custom_dir / "config_fed_client.json").write_text(json.dumps({"password": "hunter22x"}))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        findings = warn_on_potential_secrets_in_job_dir(str(tmp_path), job_name="job")

    assert findings == []
    assert not [warning for warning in record if issubclass(warning.category, PotentialSecretWarning)]
