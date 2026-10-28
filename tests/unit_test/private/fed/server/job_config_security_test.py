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

"""Remote configurations must remain inert throughout admission and deployment."""

import io
import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from nvflare.apis.utils.job_utils import convert_legacy_zipped_app_to_job
from nvflare.private.fed.app.default_app_validator import DefaultAppValidator
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator


def _archive(replacements=None):
    files = {
        "job/meta.json": json.dumps({"min_clients": 1, "deploy_map": {"app": ["server", "site-1"]}}),
        "job/app/config/": "",
        "job/app/config/config_fed_server.json": '{"format_version": 2, "components": [], "workflows": []}',
        "job/app/config/config_fed_client.json": '{"format_version": 2, "components": [], "executors": []}',
    }
    for name, value in (replacements or {}).items():
        basename = name.split(".", 1)[0]
        files = {key: data for key, data in files.items() if key.split(".", 1)[0] != basename}
        files[name] = value
    result = io.BytesIO()
    with ZipFile(result, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return result.getvalue()


@pytest.mark.parametrize(
    "path,payload",
    [
        ("meta.conf", 'include file("/synthetic/private.conf")'),
        ("meta.conf.default", "stolen=${CONFIG_SECURITY_CANARY}"),
        ("meta.yaml", "stolen: ${oc.env:CONFIG_SECURITY_CANARY}"),
        ("app/config/config_fed_server.conf", 'include url("http://127.0.0.1:1/private")'),
        ("app/config/config_fed_client.yaml", "execution_lifetime: task"),
        ("app/config/config_fed_client.conf", 'x=[a//b, include file("/synthetic/private.conf")]'),
    ],
)
def test_admission_never_invokes_rich_parsers(path, payload, monkeypatch):
    monkeypatch.setenv("CONFIG_SECURITY_CANARY", "synthetic-secret-never-returned")
    with (
        patch("pyhocon.ConfigFactory.parse_string", side_effect=AssertionError("HOCON was invoked")),
        patch("omegaconf.OmegaConf.create", side_effect=AssertionError("OmegaConf was invoked")),
        patch("pyhocon.config_parser.urlopen", side_effect=AssertionError("URL was fetched")),
    ):
        valid, error, _ = JobMetaValidator().validate("job", _archive({"job/" + path: payload}))
    assert not valid
    assert error
    assert "synthetic-secret-never-returned" not in error


@pytest.mark.parametrize("suffix", [".json", ".conf", ".conf.default"])
def test_comments_and_literal_environment_references_are_data(suffix, monkeypatch):
    monkeypatch.setenv("CONFIG_SECURITY_CANARY", "synthetic-secret-never-returned")
    literals = [
        "${CONFIG_SECURITY_CANARY}",
        "{$CONFIG_SECURITY_CANARY}",
        "${oc.env:CONFIG_SECURITY_CANARY}",
        "${secret:CONFIG_SECURITY_CANARY}",
        "${secret:file:/synthetic/private.conf}",
    ]
    meta = '{ /* metadata */ "min_clients": 1, "deploy_map": {"app": ["server", "site-1"]},\n'
    meta += '"literal": ' + json.dumps(literals) + " // retain this comment\n}"
    data = _archive({"job/meta" + suffix: meta})
    with (
        patch("nvflare.fuel.utils.wfconf.resolve_var_refs", side_effect=AssertionError("variables were resolved")),
        patch(
            "nvflare.fuel.utils.secret_utils.resolve_secret_refs", side_effect=AssertionError("secrets were resolved")
        ),
    ):
        valid, error, parsed = JobMetaValidator().validate("job", data)
    assert valid, error
    assert parsed["literal"] == literals
    with ZipFile(io.BytesIO(data)) as archive:
        assert archive.read("job/meta" + suffix).decode() == meta


def test_invalid_metadata_does_not_echo_values():
    meta = {"min_clients": "synthetic-private-value", "deploy_map": {"app": ["server", "site-1"]}}
    valid, error, _ = JobMetaValidator().validate("job", _archive({"job/meta.json": json.dumps(meta)}))
    assert not valid
    assert error == "min_clients must be an integer"


def test_yaml_metadata_override_does_not_fall_back_to_json_default():
    meta = {"min_clients": 1, "deploy_map": {"app": ["server", "site-1"]}}
    data = io.BytesIO(_archive({"job/meta.json.default": json.dumps(meta)}))
    with ZipFile(data, "a") as archive:
        archive.writestr("job/meta.yaml", "min_clients: 2\n")

    valid, error, _ = JobMetaValidator().validate("job", data.getvalue())

    assert not valid
    assert "YAML configuration" in error
    assert "convert to JSON" in error


def test_legacy_conversion_does_not_expand_metadata():
    with patch("pyhocon.ConfigFactory.parse_string", side_effect=AssertionError("HOCON was invoked")):
        with pytest.raises(ValueError):
            convert_legacy_zipped_app_to_job(_archive({"job/meta.conf": "stolen=${CONFIG_SECURITY_CANARY}"}))


def test_direct_default_validator_uses_data_only_parser(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "config_fed_client.conf").write_text('include file("/synthetic/private.conf")')
    with patch("pyhocon.ConfigFactory.parse_string", side_effect=AssertionError("HOCON was invoked")):
        with pytest.raises(ValueError):
            DefaultAppValidator("client").validate(str(tmp_path))
