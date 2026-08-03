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

"""Tests for Cell Client API bootstrap and attach-profile contracts."""

import json
import os

import pytest

from nvflare.client.cell.bootstrap import (
    ATTACH_EXECUTION_MODE,
    BOOTSTRAP_SCHEMA_VERSION,
    CELL_API_TYPE,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    get_bootstrap_client_api_type,
    read_bootstrap_config,
    write_bootstrap_config,
)

CONFIG = {
    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
    BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:56789",
    BootstrapKey.CJ_FQCN: "site-1.job-1",
    BootstrapKey.TRAINER_FQCN: "site-1.job-1.client_api_trainer_1",
    BootstrapKey.LAUNCH_TOKEN: "secret-token",
    BootstrapKey.JOB_ID: "job-1",
    BootstrapKey.SITE_NAME: "site-1",
    BootstrapKey.TASK_EXCHANGE: {"train_task_name": "train"},
}

ATTACH_CONFIG = {
    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
    BootstrapKey.EXECUTION_MODE: ATTACH_EXECUTION_MODE,
    BootstrapKey.ATTACH_ID: "trainer_a",
    BootstrapKey.SITE_NAME: "site-1",
    BootstrapKey.CJ_FQCN: "site-1.job-1",
    BootstrapKey.CONNECT_URL: "grpc://127.0.0.1:56789",
    BootstrapKey.CONNECTION_SECURITY: "clear",
    BootstrapKey.JOB_WAIT_TIMEOUT: None,
}


class TestBootstrapConfig:
    def test_write_read_round_trip(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        write_bootstrap_config(path, CONFIG)
        assert read_bootstrap_config(path) == CONFIG

    def test_file_is_owner_only(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        write_bootstrap_config(path, CONFIG)
        # the launch token must never be readable by other local users
        assert os.stat(path).st_mode & 0o777 == 0o600

    def test_overwrite_tightens_wider_preexisting_mode(self, tmp_path):
        # launch_once=False overwrites the same path per launch; a pre-existing file with
        # a wider mode (e.g. left by other tooling) must come out 0600 as well
        path = str(tmp_path / "bootstrap.json")
        with open(path, "w") as f:
            f.write("{}")
        os.chmod(path, 0o644)

        write_bootstrap_config(path, CONFIG)

        assert os.stat(path).st_mode & 0o777 == 0o600
        assert read_bootstrap_config(path) == CONFIG

    def test_write_replaces_symlink_without_touching_target(self, tmp_path):
        target = tmp_path / "unrelated.json"
        target.write_text('{"owner": "other"}')
        path = tmp_path / "bootstrap.json"
        path.symlink_to(target)

        write_bootstrap_config(str(path), CONFIG)

        assert not path.is_symlink()
        assert read_bootstrap_config(str(path)) == CONFIG
        assert target.read_text() == '{"owner": "other"}'

    def test_failed_write_preserves_existing_config_and_cleans_temp(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        write_bootstrap_config(path, CONFIG)

        with pytest.raises(TypeError):
            write_bootstrap_config(path, {"not_json_serializable": object()})

        assert read_bootstrap_config(path) == CONFIG
        assert not list(tmp_path.glob(".client_api_bootstrap-*.tmp"))

    def test_read_rejects_non_dict_content(self, tmp_path):
        path = str(tmp_path / "bootstrap.json")
        with open(path, "w") as f:
            json.dump(["not", "a", "dict"], f)
        with pytest.raises(ValueError, match="expect a JSON dict"):
            read_bootstrap_config(path)

    def test_typed_bootstrap_identifies_cell_api(self):
        assert get_bootstrap_client_api_type(CONFIG, "bootstrap.json") == CELL_API_TYPE
        assert get_bootstrap_client_api_type(ATTACH_CONFIG, "attach.json") == CELL_API_TYPE

    @pytest.mark.parametrize("field", [BootstrapKey.ATTACH_ID, BootstrapKey.SITE_NAME])
    def test_attach_profile_requires_rendezvous_fields(self, field):
        config = dict(ATTACH_CONFIG)
        del config[field]
        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            get_bootstrap_client_api_type(config, "attach.json")

    def test_attach_profile_requires_exactly_one_connection_source(self, tmp_path):
        neither = dict(ATTACH_CONFIG)
        del neither[BootstrapKey.CONNECT_URL]
        with pytest.raises(ValueError, match="exactly one"):
            get_bootstrap_client_api_type(neither, "attach.json")

        both = {
            **ATTACH_CONFIG,
            BootstrapKey.RENDEZVOUS_DIR: str(tmp_path),
        }
        with pytest.raises(ValueError, match="exactly one"):
            get_bootstrap_client_api_type(both, "attach.json")

    def test_shared_file_rendezvous_profile_is_valid(self, tmp_path):
        config = dict(ATTACH_CONFIG)
        del config[BootstrapKey.CONNECT_URL]
        del config[BootstrapKey.CJ_FQCN]
        config[BootstrapKey.RENDEZVOUS_DIR] = str(tmp_path)

        assert get_bootstrap_client_api_type(config, "attach.json") == CELL_API_TYPE

    def test_shared_file_rendezvous_requires_absolute_path_and_no_tls_material(self, tmp_path):
        config = dict(ATTACH_CONFIG)
        del config[BootstrapKey.CONNECT_URL]
        del config[BootstrapKey.CJ_FQCN]
        config[BootstrapKey.RENDEZVOUS_DIR] = "relative/path"
        with pytest.raises(ValueError, match="absolute path"):
            get_bootstrap_client_api_type(config, "attach.json")

        config[BootstrapKey.RENDEZVOUS_DIR] = str(tmp_path)
        config[BootstrapKey.CONNECTION_SECURITY] = "mtls"
        with pytest.raises(ValueError, match="supports only"):
            get_bootstrap_client_api_type(config, "attach.json")

        config[BootstrapKey.CONNECTION_SECURITY] = "clear"
        config[BootstrapKey.CA_CERT] = "/tmp/rootCA.pem"
        with pytest.raises(ValueError, match="does not use"):
            get_bootstrap_client_api_type(config, "attach.json")

    def test_shared_file_rendezvous_rejects_prebound_cj(self, tmp_path):
        config = dict(ATTACH_CONFIG)
        del config[BootstrapKey.CONNECT_URL]
        config[BootstrapKey.RENDEZVOUS_DIR] = str(tmp_path)

        with pytest.raises(ValueError, match="discovers 'cj_fqcn'"):
            get_bootstrap_client_api_type(config, "attach.json")

    @pytest.mark.parametrize(
        "cj_fqcn",
        [None, "", "site-2.job-1", "site-1", "site-1.job-1.extra", "site-1.bad job"],
    )
    def test_direct_profile_requires_job_cell_fqcn_for_the_configured_site(self, cj_fqcn):
        config = dict(ATTACH_CONFIG)
        if cj_fqcn is None:
            config.pop(BootstrapKey.CJ_FQCN)
        else:
            config[BootstrapKey.CJ_FQCN] = cj_fqcn

        with pytest.raises(ValueError, match="cj_fqcn"):
            get_bootstrap_client_api_type(config, "attach.json")

    def test_direct_clear_profile_requires_explicit_security_acknowledgement(self):
        config = dict(ATTACH_CONFIG)
        config.pop(BootstrapKey.CONNECTION_SECURITY)

        with pytest.raises(ValueError, match="explicitly set connection_security='clear'"):
            get_bootstrap_client_api_type(config, "attach.json")

    @pytest.mark.parametrize("attach_id", ["", "has.dot", "has space", "a" * 65])
    def test_attach_profile_rejects_bad_attach_id(self, attach_id):
        with pytest.raises(ValueError, match="attach_id"):
            get_bootstrap_client_api_type(
                {**ATTACH_CONFIG, BootstrapKey.ATTACH_ID: attach_id},
                "attach.json",
            )

    @pytest.mark.parametrize("value", [-1, "1", True])
    def test_attach_profile_rejects_bad_job_wait_timeout(self, value):
        with pytest.raises(ValueError, match="job_wait_timeout"):
            get_bootstrap_client_api_type(
                {**ATTACH_CONFIG, BootstrapKey.JOB_WAIT_TIMEOUT: value},
                "attach.json",
            )

    def test_secure_attach_profile_requires_ca_and_consistent_secure_mode(self):
        secure = {
            **ATTACH_CONFIG,
            BootstrapKey.CONNECT_URL: "grpcs://site.example:9000",
            BootstrapKey.CONNECTION_SECURITY: "mtls",
        }
        with pytest.raises(ValueError, match="requires field 'ca_cert'"):
            get_bootstrap_client_api_type(secure, "attach.json")
        with pytest.raises(ValueError, match="secure_mode.*disagrees"):
            get_bootstrap_client_api_type(
                {
                    **secure,
                    BootstrapKey.CA_CERT: "/workspace/startup/rootCA.pem",
                    BootstrapKey.SECURE_MODE: False,
                },
                "attach.json",
            )
        assert (
            get_bootstrap_client_api_type(
                {
                    **secure,
                    BootstrapKey.CA_CERT: "/workspace/startup/rootCA.pem",
                    BootstrapKey.SECURE_MODE: True,
                },
                "attach.json",
            )
            == CELL_API_TYPE
        )

    def test_bare_ca_tls_attach_profile_is_rejected(self):
        with pytest.raises(ValueError, match="bare-CA TLS attach is not supported"):
            get_bootstrap_client_api_type(
                {
                    **ATTACH_CONFIG,
                    BootstrapKey.CONNECT_URL: "grpcs://site.example:9000",
                    BootstrapKey.CONNECTION_SECURITY: "tls",
                    BootstrapKey.CA_CERT: "/workspace/startup/rootCA.pem",
                },
                "attach.json",
            )

    def test_untyped_legacy_config_is_not_a_bootstrap(self):
        assert get_bootstrap_client_api_type({"TASK_EXCHANGE": {}}, "legacy.json") is None

    @pytest.mark.parametrize(
        "field",
        [
            BootstrapKey.CJ_FQCN,
            BootstrapKey.TRAINER_FQCN,
            BootstrapKey.JOB_ID,
            BootstrapKey.SITE_NAME,
            BootstrapKey.CONNECT_URL,
            BootstrapKey.LAUNCH_TOKEN,
        ],
    )
    def test_typed_bootstrap_rejects_missing_required_string_field(self, field):
        config = dict(CONFIG)
        del config[field]

        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            get_bootstrap_client_api_type(config, "bootstrap.json")

    @pytest.mark.parametrize(
        "field",
        [
            BootstrapKey.CJ_FQCN,
            BootstrapKey.TRAINER_FQCN,
            BootstrapKey.JOB_ID,
            BootstrapKey.SITE_NAME,
            BootstrapKey.CONNECT_URL,
            BootstrapKey.LAUNCH_TOKEN,
        ],
    )
    @pytest.mark.parametrize("invalid_value", [None, "", "   ", 123])
    def test_typed_bootstrap_rejects_invalid_required_string_field(self, field, invalid_value):
        config = {**CONFIG, field: invalid_value}

        with pytest.raises(ValueError, match=f"field '{field}' must be a non-empty string"):
            get_bootstrap_client_api_type(config, "bootstrap.json")

    @pytest.mark.parametrize("base_config", [CONFIG, ATTACH_CONFIG], ids=["external_process", "attach"])
    def test_typed_bootstrap_rejects_invalid_secure_mode(self, base_config):
        config = {**base_config, BootstrapKey.SECURE_MODE: "true"}

        with pytest.raises(ValueError, match="field 'secure_mode' must be a bool"):
            get_bootstrap_client_api_type(config, "bootstrap.json")

    @pytest.mark.parametrize(
        "config,match",
        [
            (
                {BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE},
                "missing required field 'schema_version'",
            ),
            (
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
                    BootstrapKey.EXECUTION_MODE: "bogus",
                },
                "unsupported Client API bootstrap execution_mode 'bogus'",
            ),
            (
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION + 1,
                    BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
                },
                "unsupported Client API bootstrap schema_version 2",
            ),
        ],
    )
    def test_typed_bootstrap_rejects_incomplete_or_unsupported_envelope(self, config, match):
        with pytest.raises(ValueError, match=match):
            get_bootstrap_client_api_type(config, "bootstrap.json")
