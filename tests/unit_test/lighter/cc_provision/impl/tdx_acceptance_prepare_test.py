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

from nvflare.lighter.cc_provision.impl.coco import validate_coco_config

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
    return dict(
        output=tmp_path / "run",
        run_id="reviewed-run",
        base_image="example/base@sha256:" + "a" * 64,
        as_key=key,
        as_key_sha256=hashlib.sha256(public).hexdigest(),
        mrtd="b" * 96,
        platform=platform,
        server="server.example.com",
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
        for name in required:
            config = yaml.safe_load((output / topology / f"cc_{name}.yml").read_text())
            validate_coco_config(config)
            assert set(config["cc_issuers"][0]["args"]["workload_constraints"]) == required
            assert config["cc_gpu"] == "none"
            assert "init_data" not in config["cc_issuers"][0]["args"]["workload_constraints"][name]
            assert (output / topology / name / "tdx_acceptance.py").is_file()
    with pytest.raises(FileExistsError):
        load("prepare").prepare(**inputs)


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


@pytest.mark.parametrize("protected", [False, True])
def test_observer_is_verifier_only_before_signing(tmp_path, protected):
    server = SimpleNamespace(name="server.example.com")
    observer = SimpleNamespace(name="site-observer")
    source = tmp_path / server.name
    source.mkdir()
    (source / "resources.json.default").write_text(json.dumps({"components": [], "preserved": "value"}))
    (tmp_path / observer.name).mkdir()
    authorizer = dict(
        audience="nvflare-coco:run", trustee_public_key="public", workload_constraints={"site-1": {"cpu_tee": "tdx"}}
    )
    if protected:
        authorizer.update(site_name="server", token_url="http://127.0.0.1:8006/aa/token", retry_max_attempts=10)
    manager = dict(
        cc_issuers_conf=[{"issuer_id": "coco_authorizer"}] if protected else [],
        require_site_binding=True,
        required_site_verifier_ids={"site-1": ["coco_authorizer"]},
    )
    for name, args in (("coco_authorizer", authorizer), ("cc_manager", manager)):
        (source / f"{name}__p_resources.json").write_text(json.dumps({"components": [{"args": args}]}))
    project = SimpleNamespace(get_clients=lambda: [observer], get_server=lambda: server)
    ctx = SimpleNamespace(get_local_dir=lambda p: tmp_path / p.name)
    load("observer_builder").ObserverBuilder().build(project, ctx)
    result = json.loads((tmp_path / observer.name / "coco_authorizer__p_resources.json").read_text())["components"][0][
        "args"
    ]
    assert result == {k: v for k, v in authorizer.items() if k not in {"site_name", "token_url", "retry_max_attempts"}}
    result = json.loads((tmp_path / observer.name / "cc_manager__p_resources.json").read_text())["components"][0][
        "args"
    ]
    assert result["cc_issuers_conf"] == []
    assert result["require_site_binding"]
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
    # Leave CoCoPackager disabled: this proves signed private-kit generation,
    # not encrypted image packaging or release readiness.
    ctx = Provisioner(str(output / topology / "offline-workspace"), pipeline).provision(project)
    assert ctx.get(CtxKey.PROVISION_SUCCESS) is True, ctx.get_errors()
    result = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    protected = {"site-1", "site-2"} | ({"server"} if topology == "b" else set())
    expected_map = {site: ["coco_authorizer"] for site in protected}
    for identity in (inputs["server"], "site-1", "site-2", "site-observer"):
        kit = result / identity
        local = kit / "local"
        manager = json.loads((local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = json.loads((local / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
        assert manager["required_site_verifier_ids"] == expected_map
        assert set(manager["cc_enabled_sites"]) == protected
        assert manager["require_site_binding"] is True
        assert manager["verify_frequency"] == 120
        assert manager["registration_token_timeout"] == 300
        assert manager["refresh_token_timeout"] == 30
        assert manager["get_token_request_timeout"] == 45
        assert set(authorizer["workload_constraints"]) == protected
        assert authorizer["audience"] == "nvflare-coco:" + project.name
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
            assert manager["cc_issuers_conf"] == [{"issuer_id": "coco_authorizer", "token_expiration": 300}]
            assert verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
            # Observer/server public configuration must be covered before any
            # acceptance of modified local resources on protected participants.
            resources = local / "coco_authorizer__p_resources.json"
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
