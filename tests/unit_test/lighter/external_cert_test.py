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
from pathlib import Path

import pytest

from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner


def _provision(tmp_path: Path, external_cert=True, external_job_ca=False) -> Path:
    project = prepare_project(
        {
            "api_version": 3,
            "name": "external-certs",
            "description": "test",
            "participants": [
                {
                    "type": "server",
                    "name": "server1",
                    "org": "org",
                    "external_cert": external_cert,
                    "external_job_ca": external_job_ca,
                },
                {"type": "client", "name": "site-1", "org": "org", "external_cert": external_cert},
            ],
        }
    )
    context = Provisioner(
        str(tmp_path),
        [WorkspaceBuilder(), StaticFileBuilder(scheme="grpc"), CertBuilder(), SignatureBuilder()],
    ).provision(project)

    assert not context.get(CtxKey.BUILD_ERROR)
    return Path(context[CtxKey.CURRENT_PROD_DIR])


def test_external_workload_cert_kit_omits_leaf_credentials_but_keeps_runtime_config(tmp_path):
    prod_dir = _provision(tmp_path)
    server_startup = prod_dir / "server1" / "startup"
    client_startup = prod_dir / "site-1" / "startup"

    for startup_dir, cert_name, key_name, config_name in (
        (server_startup, "server.crt", "server.key", "fed_server.json"),
        (client_startup, "client.crt", "client.key", "fed_client.json"),
    ):
        assert (startup_dir / "rootCA.pem").is_file()
        assert (startup_dir / config_name).is_file()
        assert not (startup_dir / cert_name).exists()
        assert not (startup_dir / key_name).exists()

    server_config = json.loads((server_startup / "fed_server.json").read_text())
    client_config = json.loads((client_startup / "fed_client.json").read_text())
    assert server_config["servers"][0]["ssl_cert"] == "server.crt"
    assert server_config["servers"][0]["ssl_private_key"] == "server.key"
    assert client_config["client"]["ssl_cert"] == "client.crt"
    assert client_config["client"]["ssl_private_key"] == "client.key"
    assert "external_cert" not in server_config["servers"][0]
    assert "external_cert" not in client_config["client"]
    assert "external_cert: true" in (prod_dir / "server1" / "readme.txt").read_text()
    assert "external_cert: true" in (prod_dir / "site-1" / "readme.txt").read_text()


@pytest.mark.parametrize("external_cert", [False, True])
@pytest.mark.parametrize("external_job_ca", [False, True])
def test_endpoint_and_job_ca_ownership_are_independent(tmp_path, external_cert, external_job_ca):
    prod = _provision(tmp_path, external_cert=external_cert, external_job_ca=external_job_ca)
    startup = prod / "server1" / "startup"
    assert (startup / "rootCA.pem").is_file()
    for suffix in ("crt", "key"):
        assert (startup / f"server.{suffix}").exists() is not external_cert
        assert (startup / f"job_ca.{suffix}").exists() is not external_job_ca
    state = json.loads(next(tmp_path.rglob("cert.json")).read_text())
    assert ("job_ca.external-certs" in state) is not external_job_ca


def test_external_job_ca_does_not_reuse_or_overwrite_provisioned_material(tmp_path):
    original = _provision(tmp_path)
    original_key = (original / "server1" / "startup" / "job_ca.key").read_bytes()
    external = _provision(tmp_path, external_job_ca=True)
    assert not (external / "server1" / "startup" / "job_ca.key").exists()
    assert not (external / "server1" / "startup" / "job_ca.crt").exists()
    assert (original / "server1" / "startup" / "job_ca.key").read_bytes() == original_key


@pytest.mark.parametrize("role,value", [("client", True), ("relay", True), ("server", "true"), ("server", 1)])
def test_external_job_ca_requires_server_and_boolean(role, value):
    with pytest.raises(ValueError, match="external_job_ca"):
        prepare_project(
            {
                "api_version": 3,
                "name": "test",
                "description": "test",
                "participants": [{"type": role, "name": "site1", "org": "org", "external_job_ca": value}],
            }
        )
