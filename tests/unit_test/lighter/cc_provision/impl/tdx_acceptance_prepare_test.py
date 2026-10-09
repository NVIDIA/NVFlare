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

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.config import load_participant_config, load_project_config

ROOT = Path(__file__).resolve().parents[5]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/devops/coco/acceptance" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    public = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    key = tmp_path / "as.pem"
    key.write_bytes(public)
    platform = tmp_path / "platform.env"
    platform.write_text("# externally approved platform configuration\n")
    for name in ("trustee-ca.pem", "admin.jwt", "registry-ca.pem", "username", "password"):
        (tmp_path / name).write_text("fixture")
    command = tmp_path / "build"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o700)
    cc_project = tmp_path / "cc_project.yml"
    cc_project.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "attestation_services": {
                    "trustee": {
                        "type": "trustee",
                        "kbs_endpoint": "https://trustee.example.com:8443",
                        "ca_cert_file": "trustee-ca.pem",
                        "admin_token_file": "admin.jwt",
                        "attestation_token_endpoint": "http://127.0.0.1:8006/aa/token",
                        "attestation_signing_public_key_file": "as.pem",
                        "token_expiration_seconds": 300,
                        "check_frequency_seconds": 120,
                        "registration_token_timeout_seconds": 300,
                        "refresh_token_timeout_seconds": 30,
                        "get_token_request_timeout_seconds": 45,
                    }
                },
                "container_registries": {
                    "workloads": {
                        "endpoint": "registry.example.com:5000",
                        "ca_cert_file": "registry-ca.pem",
                        "publisher_username_file": "username",
                        "publisher_password_file": "password",
                    }
                },
                "build_tools": {"coco": {"build_command": "build"}},
            },
            sort_keys=False,
        )
    )
    return dict(
        output=tmp_path / "run",
        run_id="reviewed-run",
        base_image="example/base@sha256:" + "a" * 64,
        as_key=key,
        as_key_sha256=hashlib.sha256(public).hexdigest(),
        mrtd="b" * 96,
        platform=platform,
        server="server.example.com",
        cc_project=cc_project,
    )


def test_private_preparation_and_complete_constraints(inputs):
    output = load("prepare").prepare(**inputs)
    assert output.stat().st_mode & 0o777 == 0o700
    for topology in ("a", "b"):
        project = yaml.safe_load((output / topology / "project.yaml").read_text())
        assert len(project["participants"]) == 5
        assert project["builders"][-2]["path"] == "observer_builder.ObserverBuilder"
        observer = next(p for p in project["participants"] if p["name"] == "site-observer")
        assert "cc_config" not in observer
        required = {"site-1", "site-2"} | ({"server"} if topology == "b" else set())
        project_config = load_project_config(output / topology / "cc_project.yml")
        constraints = project_config["attestation_services"]["trustee"]["workload_constraints"]
        assert set(constraints) == required
        for name in required:
            config = load_participant_config(output / topology / f"cc_{name}.yml", project_config)
            assert config["gpu_tee"].value == "none"
            assert "init_data" not in constraints[name]
            assert (output / topology / name / "tdx_acceptance.py").is_file()
    with pytest.raises(FileExistsError):
        load("prepare").prepare(**inputs)


def test_preparation_preserves_normalized_project_paths(inputs, monkeypatch):
    source_root = Path(inputs["cc_project"]).parent
    home = source_root / "home"
    credentials = home / "credentials"
    credentials.mkdir(parents=True)
    for name in ("trustee-ca.pem", "admin.jwt", "registry-ca.pem", "username", "password"):
        (credentials / name).write_text("fixture")
    command = home / "bin" / "build"
    command.parent.mkdir()
    command.write_text("#!/bin/sh\n")
    command.chmod(0o700)
    approval = source_root / "approval.pub"
    approval.write_text("fixture")
    builder = source_root / "builder"
    builder.mkdir()
    (builder / "cvmctl").write_text("#!/bin/sh\n")
    (builder / "cvmctl").chmod(0o700)

    project_config = yaml.safe_load(Path(inputs["cc_project"]).read_text())
    trustee = project_config["attestation_services"]["trustee"]
    trustee["ca_cert_file"] = "~/credentials/trustee-ca.pem"
    trustee["admin_token_file"] = "~/credentials/admin.jwt"
    registry = project_config["container_registries"]["workloads"]
    for field, name in (
        ("ca_cert_file", "registry-ca.pem"),
        ("publisher_username_file", "username"),
        ("publisher_password_file", "password"),
    ):
        registry[field] = f"~/credentials/{name}"
    project_config["approval"] = {"public_key_files": ["approval.pub"]}
    project_config["build_tools"] = {
        "coco": {"build_command": "~/bin/build"},
        "bare_metal_cvm": {"cvm_builder_dir": "builder", "output_root": "external-output"},
    }
    Path(inputs["cc_project"]).write_text(yaml.safe_dump(project_config, sort_keys=False))
    monkeypatch.setenv("HOME", str(home))

    output = load("prepare").prepare(**inputs)

    for topology in ("a", "b"):
        generated = load_project_config(output / topology / "cc_project.yml")
        service = generated["attestation_services"]["trustee"]
        assert service["ca_cert_file"] == str(credentials / "trustee-ca.pem")
        assert service["admin_token_file"] == str(credentials / "admin.jwt")
        assert service["attestation_signing_public_key_file"] == str(output / topology / "trustee-as-public.pem")
        generated_registry = generated["container_registries"]["workloads"]
        assert generated_registry["ca_cert_file"] == str(credentials / "registry-ca.pem")
        assert generated_registry["publisher_username_file"] == str(credentials / "username")
        assert generated_registry["publisher_password_file"] == str(credentials / "password")
        assert generated["approval"]["public_key_files"] == [str(approval)]
        assert generated["build_tools"]["coco"]["build_command"] == str(command)
        assert generated["build_tools"]["bare_metal_cvm"]["cvm_builder_dir"] == str(builder)
        assert generated["build_tools"]["bare_metal_cvm"]["output_root"] == str(source_root / "external-output")


@pytest.mark.parametrize("topology", ["a", "b"])
def test_private_application_can_be_read_by_approved_guest_identity(inputs, topology):
    """Private source stays 0600; Docker must transfer ownership to the guest UID."""
    output = load("prepare").prepare(**inputs)
    protected = {"site-1", "site-2"} | ({"server"} if topology == "b" else set())
    reviewed = (ROOT / "examples/devops/coco/acceptance/application/tdx_acceptance.py").read_bytes()
    for site in protected:
        context = output / topology / site
        source = context / "tdx_acceptance.py"
        assert source.stat().st_mode & 0o777 == 0o600
        assert source.read_bytes() == reviewed
        lines = (context / "Dockerfile").read_text().splitlines()
        assert lines[0] == "FROM " + inputs["base_image"]
        copies = [line for line in lines if line.startswith("COPY ")]
        assert copies == ["COPY --chown=65532:65532 tdx_acceptance.py /local/custom/tdx_acceptance.py"]


@pytest.mark.parametrize(
    "field,value",
    [("as_key_sha256", "0" * 64), ("mrtd", "unapproved"), ("base_image", "image:latest"), ("run_id", "../escape")],
)
def test_reject_untrusted_inputs(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        load("prepare").prepare(**inputs)
    assert not inputs["output"].exists()


def test_observer_builder_only_authorizes_baked_server_classes(tmp_path):
    server = SimpleNamespace(name="server.example.com")
    source = tmp_path / server.name
    source.mkdir()
    (source / "resources.json.default").write_text(json.dumps({"components": [], "preserved": "value"}))
    project = SimpleNamespace(get_server=lambda: server)
    ctx = SimpleNamespace(get_local_dir=lambda p: tmp_path / p.name)
    load("observer_builder").ObserverBuilder().build(project, ctx)
    resources = json.loads((source / "resources.json.default").read_text())
    assert resources["preserved"] == "value"
    assert resources["components"] == []
    assert resources["class_allow_list"].count("tdx_acceptance.AcceptanceController") == 1
    assert resources["class_allow_list"].count("tdx_acceptance.AcceptanceExecutor") == 1


@pytest.mark.parametrize(
    "policy",
    [
        {"class_allow_list": ["*"]},
        {"class_allow_list": ["tdx_acceptance.*"]},
        {"class_allow_list": ["tdx_acceptance."]},
        {"class_allow_list": ["nvflare.app_common."]},
        {"class_allow_list": ["tdx_acceptance"]},
        {"class_allow_list": [""]},
        {"class_allow_list": [None]},
        {"class_allow_list": "tdx_acceptance.AcceptanceController"},
        {"class_list_enforcement_mode": "warn"},
    ],
)
def test_baked_application_keeps_component_enforcement(tmp_path, policy):
    resources = tmp_path / "resources.json.default"
    original = json.dumps(policy)
    resources.write_text(original)
    with pytest.raises(ValueError):
        load("observer_builder").ObserverBuilder._allow_baked_application(resources)
    assert resources.read_text() == original


def test_baked_application_preserves_explicit_policy_and_is_idempotent(tmp_path):
    builder = load("observer_builder").ObserverBuilder
    resources = tmp_path / "resources.json.default"
    existing = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
    resources.write_text(json.dumps({"class_allow_list": [existing], "class_list_enforcement_mode": "enforce"}))
    builder._allow_baked_application(resources)
    first = resources.read_bytes()
    builder._allow_baked_application(resources)
    assert resources.read_bytes() == first
    assert json.loads(first)["class_allow_list"] == [
        existing,
        "tdx_acceptance.AcceptanceController",
        "tdx_acceptance.AcceptanceExecutor",
    ]


@pytest.mark.parametrize("topology", ["a", "b"])
def test_offline_real_builders_produce_verified_kits_and_observer(inputs, monkeypatch, topology):
    """Exercise real provision parsing/finalization; no image build or trust-service request."""
    from nvflare.lighter.constants import CtxKey, ProvFileName
    from nvflare.lighter.prov_utils import prepare_builders
    from nvflare.lighter.provision import prepare_project
    from nvflare.lighter.provisioner import Provisioner
    from nvflare.lighter.utils import verify_folder_signature

    output = load("prepare").prepare(**inputs)
    source = output / topology / "project.yaml"
    definition = yaml.safe_load(source.read_text())
    monkeypatch.syspath_prepend(str(ROOT / "examples/devops/coco/acceptance"))
    project = prepare_project(definition, project_file=str(source))
    pipeline = prepare_builders(definition)
    # Leave CCPackager disabled: this proves signed private-kit generation,
    # not encrypted image packaging or release readiness.
    ctx = Provisioner(str(output / topology / "offline-workspace"), pipeline).provision(project)
    assert ctx.get(CtxKey.PROVISION_SUCCESS) is True, ctx.get_errors()
    result = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    protected = {"site-1", "site-2"} | ({"server"} if topology == "b" else set())
    expected_map = {site: ["trustee_authorizer"] for site in protected}
    for identity in (inputs["server"], "site-1", "site-2", "site-observer"):
        kit = result / identity
        local = kit / "local"
        manager = json.loads((local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = json.loads((local / "trustee_authorizer__p_resources.json").read_text())["components"][0]["args"]
        assert manager["required_site_verifier_ids"] == expected_map
        assert set(manager["cc_enabled_sites"]) == protected
        assert manager["require_site_binding"] is True
        assert manager["verify_frequency"] == 120
        assert manager["registration_token_timeout"] == 300
        assert manager["refresh_token_timeout"] == 30
        assert manager["get_token_request_timeout"] == 45
        assert set(authorizer["workload_constraints"]) == protected
        assert authorizer["audience"] == "nvflare-trustee:" + project.name
        logical_identity = "server" if identity == inputs["server"] else identity
        if identity == inputs["server"]:
            from nvflare.apis.fl_exception import UnsafeComponentError
            from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer

            policy = ComponentPathAuthorizer()
            workspace = SimpleNamespace(get_resources_file_path=lambda: str(local / "resources.json.default"))
            for name in ("tdx_acceptance.AcceptanceController", "tdx_acceptance.AcceptanceExecutor"):
                policy.authorize_component_config({"path": name}, workspace=workspace)
            for name in ("tdx_acceptance.Unapproved", "tdx_acceptance.AcceptanceControllerExtra"):
                with pytest.raises(UnsafeComponentError):
                    policy.authorize_component_config({"path": name}, workspace=workspace)
        if logical_identity in protected:
            assert authorizer["site_name"] == logical_identity
            assert manager["cc_issuers_conf"] == [{"issuer_id": "trustee_authorizer", "token_expiration": 300}]
            assert verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
            # Observer/server public configuration must be covered before any
            # acceptance of modified local resources on protected participants.
            resources = local / "trustee_authorizer__p_resources.json"
            resources.write_text(resources.read_text() + " ")
            assert not verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
        else:
            assert manager["cc_issuers_conf"] == []
            assert "site_name" not in authorizer
            assert "token_url" not in authorizer
            assert not any(key.startswith("retry_") for key in authorizer)
    assert (result / "admin@example.com/startup").is_dir()
