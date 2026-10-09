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

"""Exercise the normal mixed example's real builders without publishing images."""

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import verify_folder_signature

ROOT = Path(__file__).resolve().parents[5]
EXAMPLE = ROOT / "examples/devops/coco/provision/mixed-tdx-snp"


def load_builder():
    spec = importlib.util.spec_from_file_location("mixed_validation_builder", EXAMPLE / "validation_builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_builders_sign_normal_mixed_kits(tmp_path, monkeypatch):
    private = tmp_path / "mixed"
    shutil.copytree(EXAMPLE, private)
    public = ec.generate_private_key(ec.SECP256R1()).public_key()
    (private / "trustee-as-public.pem").write_bytes(
        public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    common = yaml.safe_load((private / "cc_project.yml").read_text())
    for name in ("trustee-ca.pem", "admin.jwt", "registry-ca.pem", "username", "password"):
        (private / name).write_text("fixture")
    command = private / "build"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o700)
    trustee = common["attestation_services"]["trustee"]
    trustee["ca_cert_file"] = "trustee-ca.pem"
    trustee["admin_token_file"] = "admin.jwt"
    registry = common["container_registries"]["coco_workloads"]
    registry["ca_cert_file"] = "registry-ca.pem"
    registry["publisher_username_file"] = "username"
    registry["publisher_password_file"] = "password"
    common["build_tools"]["coco"]["build_command"] = "build"
    (private / "cc_project.yml").write_text(yaml.safe_dump(common, sort_keys=False))
    (private / "platform.env").write_text("# fixture\n")
    for name in ("site-1", "site-2"):
        participant = private / f"cc_{name}.yml"
        config = yaml.safe_load(participant.read_text())
        config["coco"]["platform_config_file"] = "platform.env"
        participant.write_text(yaml.safe_dump(config, sort_keys=False))
    source = private / "project.yaml"
    definition = yaml.safe_load(source.read_text())
    monkeypatch.syspath_prepend(str(private))
    project = prepare_project(definition, project_file=str(source))
    # Disable packaging only in this offline test; real provisioning uses CCPackager.
    ctx = Provisioner(str(tmp_path / "workspace"), prepare_builders(definition)).provision(project)
    assert ctx.get(CtxKey.PROVISION_SUCCESS) is True, ctx.get_errors()
    result = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    assert {p.name for p in result.iterdir() if p.is_dir()} == {
        "server.example.com",
        "site-1",
        "site-2",
        "admin@example.com",
    }
    constraints = {
        "site-1": {"cpu_tee": "tdx", "gpu_required": False},
        "site-2": {"cpu_tee": "snp", "gpu_required": True},
    }
    for name in ("server.example.com", "site-1", "site-2"):
        kit = result / name
        manager = json.loads((kit / "local/cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = json.loads((kit / "local/trustee_authorizer__p_resources.json").read_text())["components"][0][
            "args"
        ]
        assert set(manager["cc_enabled_sites"]) == {"site-1", "site-2"}
        assert manager["required_site_verifier_ids"] == {site: ["trustee_authorizer"] for site in constraints}
        assert manager["require_site_binding"] is True
        assert manager["verify_frequency"] == 120
        assert manager["registration_token_timeout"] == 300
        assert manager["refresh_token_timeout"] == 30
        assert manager["get_token_request_timeout"] == 45
        assert authorizer["workload_constraints"] == constraints
        assert authorizer["audience"] == "nvflare-trustee:mixed_coco_project"
        resources = json.loads((kit / "local/resources.json.default").read_text())
        for cls in load_builder().APPLICATION_CLASSES:
            assert resources["class_allow_list"].count(cls) == 1
        if name == "server.example.com":
            assert manager["cc_issuers_conf"] == []
            assert "site_name" not in authorizer
        else:
            assert authorizer["site_name"] == name
            assert manager["cc_issuers_conf"] == [{"issuer_id": "trustee_authorizer", "token_expiration": 300}]
            assert (kit / "startup/sub_start.sh").read_text().startswith("#!/usr/bin/env bash\nexec >/dev/null 2>&1\n")
        if name != "server.example.com":
            assert verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
    protected = result / "site-1"
    resources = protected / "local/trustee_authorizer__p_resources.json"
    resources.write_text(resources.read_text() + " ")
    assert not verify_folder_signature(
        str(protected),
        str(protected / "startup/rootCA.pem"),
        single_signer=True,
        signature_file=ProvFileName.SIGNATURE_JSON,
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"class_allow_list": ["*"]},
        {"class_allow_list": ["my_application.*"]},
        {"class_allow_list": [True]},
        {"class_allow_list": "tdx_acceptance.AcceptanceController"},
        {"class_list_enforcement_mode": "warn"},
    ],
)
def test_validation_builder_rejects_weakened_component_policy(tmp_path, policy):
    path = tmp_path / "resources.json.default"
    original = json.dumps(policy)
    path.write_text(original)
    project = SimpleNamespace(get_server=lambda: object())
    ctx = SimpleNamespace(get_local_dir=lambda participant: tmp_path)
    with pytest.raises(ValueError, match="enforced explicit"):
        load_builder().ValidationBuilder().build(project, ctx)
    assert path.read_text() == original


def test_validation_builder_preserves_existing_resources_and_is_idempotent(tmp_path):
    path = tmp_path / "resources.json.default"
    path.write_text(json.dumps({"components": [{"id": "preserved"}], "class_allow_list": ["my_app.Other"]}))
    project = SimpleNamespace(get_server=lambda: object())
    ctx = SimpleNamespace(get_local_dir=lambda participant: tmp_path)
    builder = load_builder().ValidationBuilder()
    builder.build(project, ctx)
    builder.build(project, ctx)
    resources = json.loads(path.read_text())
    assert resources["components"] == [{"id": "preserved"}]
    assert resources["class_allow_list"] == ["my_app.Other", *load_builder().APPLICATION_CLASSES]
