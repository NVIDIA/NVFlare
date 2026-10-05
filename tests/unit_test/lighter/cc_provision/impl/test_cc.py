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
import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.config import load_participant_config, load_project_config
from nvflare.lighter.cc_provision.deployment import CPUTEE, GPUTEE, CCDeploymentMode, WorkloadSourceType
from nvflare.lighter.cc_provision.impl.bare_metal_cvm import BareMetalCVMDeployment
from nvflare.lighter.cc_provision.impl.cc import CC_PACKAGER_PATH, CCBuilder
from nvflare.lighter.cc_provision.impl.cc_packager import CCPackager
from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import verify_folder_signature


def _write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def _azure_project(tmp_path, participant=None, project_config=None):
    participant = participant or {
        "schema_version": 1,
        "cc_deployment_mode": "azure_cc",
        "cpu_tee": "amd_sev_snp",
        "gpu_tee": "none",
        "attestation": {"service": "azure_maa"},
        "class_allow_list": ["example.components.ReviewedExecutor"],
        "workload": {"source": {"type": "external"}},
        "azure_cc": {"deployment_target": "confidential_vm"},
    }
    project_config = project_config or {
        "schema_version": 1,
        "attestation_services": {
            "azure_maa": {
                "type": "azure_maa",
                "endpoint": "https://sharedeus2.eus2.attest.azure.net",
                "token_expiration_seconds": 100,
                "check_frequency_seconds": 60,
            }
        },
    }
    _write(tmp_path / "cc_project.yml", project_config)
    _write(tmp_path / "cc_server.yml", participant)
    project_dict = {
        "api_version": 3,
        "name": "unified",
        "cc_project_config": "cc_project.yml",
        "participants": [
            {
                "type": "server",
                "name": "server.example.com",
                "org": "example",
                "cc_config": "cc_server.yml",
            }
        ],
        "packager": {"path": CC_PACKAGER_PATH},
    }
    project = prepare_project(project_dict, project_file=tmp_path / "project.yml")
    return project


def _resources(ctx, participant):
    local = Path(ctx.get_local_dir(participant))
    local.mkdir(parents=True, exist_ok=True)
    path = local / ProvFileName.RESOURCES_JSON_DEFAULT
    path.write_text(json.dumps({"format_version": 2, "class_allow_list": [], "components": []}))
    return path


def _coco_project(tmp_path):
    for name in ("ca.pem", "admin.jwt", "registry-ca.pem", "username", "password"):
        (tmp_path / name).write_text("fixture")
    public = ec.generate_private_key(ec.SECP256R1()).public_key()
    (tmp_path / "as.pem").write_bytes(
        public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    command = tmp_path / "build"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o700)
    context = tmp_path / "site"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM reviewed-base\n")
    admin = tmp_path / "admin"
    admin.mkdir()
    (admin / "platform.env").write_text("# fixture\n")
    _write(
        tmp_path / "cc_project.yml",
        {
            "schema_version": 1,
            "attestation_services": {
                "trustee": {
                    "type": "trustee",
                    "kbs_endpoint": "https://trustee.example.com:8443",
                    "ca_cert_file": "ca.pem",
                    "admin_token_file": "admin.jwt",
                    "attestation_token_endpoint": "http://127.0.0.1:8006/aa/token",
                    "attestation_signing_public_key_file": "as.pem",
                    "token_expiration_seconds": 300,
                    "check_frequency_seconds": 120,
                    "registration_token_timeout_seconds": 300,
                    "refresh_token_timeout_seconds": 30,
                    "get_token_request_timeout_seconds": 45,
                    "proof_iat_leeway_seconds": 90,
                    "workload_constraints": {
                        "site-1": {
                            "cpu_tee": "snp",
                            "init_data": "a" * 64,
                        }
                    },
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
    )
    _write(
        tmp_path / "cc_site.yml",
        {
            "schema_version": 1,
            "cc_deployment_mode": "coco",
            "cpu_tee": "amd_sev_snp",
            "gpu_tee": "nvidia_cc",
            "attestation": {"service": "trustee"},
            "workload": {"source": {"type": "docker_build", "context": "site", "dockerfile": "Dockerfile"}},
            "coco": {
                "release_name": "site-v1",
                "registry": "workloads",
                "registry_repository": "workloads/site",
                "platform_config_file": "admin/platform.env",
            },
        },
    )
    project_dict = {
        "api_version": 3,
        "name": "coco-unified",
        "cc_project_config": "cc_project.yml",
        "participants": [
            {"type": "server", "name": "server.example.com", "org": "example"},
            {"type": "client", "name": "site-1", "org": "example", "cc_config": "cc_site.yml"},
        ],
        "packager": {"path": CC_PACKAGER_PATH},
    }
    return prepare_project(project_dict, project_file=tmp_path / "project.yml")


def test_unified_azure_config_creates_immutable_plan(tmp_path):
    project = _azure_project(tmp_path)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    builder = CCBuilder()

    builder.initialize(project, ctx)

    plan = ctx[CtxKey.CC_DEPLOYMENT_PLANS]["server.example.com"]
    assert plan.mode is CCDeploymentMode.AZURE_CC
    assert plan.cpu_tee is CPUTEE.AMD_SEV_SNP
    assert plan.gpu_tee is GPUTEE.NONE
    assert plan.workload_source.source_type is WorkloadSourceType.EXTERNAL
    with pytest.raises(Exception):
        plan.mode = CCDeploymentMode.COCO
    with pytest.raises(TypeError):
        plan.internal["site_name"] = "other-site"


def test_builder_writes_azure_authorizer_manager_and_allow_list(tmp_path):
    project = _azure_project(tmp_path)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    server = project.get_server()
    resources = _resources(ctx, server)
    builder = CCBuilder()

    builder.initialize(project, ctx)
    builder.build(project, ctx)

    local = Path(ctx.get_local_dir(server))
    manager = json.loads((local / "cc_manager__p_resources.json").read_text())["components"][0]
    authorizer_id = manager["args"]["required_site_verifier_ids"]["server"][0]
    authorizer = json.loads((local / f"{authorizer_id}__p_resources.json").read_text())["components"][0]
    assert authorizer["args"]["maa_endpoint"] == "sharedeus2.eus2.attest.azure.net"
    assert authorizer_id.startswith("az_cvm_authorizer_")
    assert json.loads(resources.read_text())["class_allow_list"] == ["example.components.ReviewedExecutor"]


def test_azure_participants_with_different_maa_endpoints_get_distinct_verifiers(tmp_path):
    services = {
        "east": {
            "type": "azure_maa",
            "endpoint": "https://sharedeus2.eus2.attest.azure.net",
            "token_expiration_seconds": 100,
            "check_frequency_seconds": 60,
        },
        "west": {
            "type": "azure_maa",
            "endpoint": "https://sharedwus2.wus2.attest.azure.net",
            "token_expiration_seconds": 100,
            "check_frequency_seconds": 60,
        },
    }
    _write(tmp_path / "cc_project.yml", {"schema_version": 1, "attestation_services": services})
    participant = {
        "schema_version": 1,
        "cc_deployment_mode": "azure_cc",
        "cpu_tee": "amd_sev_snp",
        "gpu_tee": "none",
        "workload": {"source": {"type": "external"}},
        "azure_cc": {"deployment_target": "confidential_vm"},
    }
    for name, service in (("server", "east"), ("site", "west")):
        _write(tmp_path / f"cc_{name}.yml", {**participant, "attestation": {"service": service}})
    project = prepare_project(
        {
            "api_version": 3,
            "name": "azure-multi-maa",
            "cc_project_config": "cc_project.yml",
            "participants": [
                {
                    "type": "server",
                    "name": "server.example.com",
                    "org": "example",
                    "cc_config": "cc_server.yml",
                },
                {"type": "client", "name": "site-1", "org": "example", "cc_config": "cc_site.yml"},
            ],
            "packager": {"path": CC_PACKAGER_PATH},
        },
        project_file=tmp_path / "project.yml",
    )
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    for party in (project.get_server(), *project.get_clients()):
        _resources(ctx, party)
    builder = CCBuilder()

    builder.initialize(project, ctx)
    builder.build(project, ctx)

    manager_path = Path(ctx.get_local_dir(project.get_server())) / "cc_manager__p_resources.json"
    manager = json.loads(manager_path.read_text())["components"][0]
    required = manager["args"]["required_site_verifier_ids"]
    east_id = required["server"][0]
    west_id = required["site-1"][0]
    assert east_id != west_id
    assert manager["args"]["cc_verifier_ids"] == [east_id, west_id]
    for authorizer_id, endpoint in (
        (east_id, "sharedeus2.eus2.attest.azure.net"),
        (west_id, "sharedwus2.wus2.attest.azure.net"),
    ):
        path = Path(ctx.get_local_dir(project.get_server())) / f"{authorizer_id}__p_resources.json"
        authorizer = json.loads(path.read_text())["components"][0]
        assert authorizer["args"]["maa_endpoint"] == endpoint


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("cc_deployment_mode", "azure_cvm", "cc_deployment_mode"),
        ("cpu_tee", "intel_tdx", "currently supports only"),
        ("gpu_tee", "nvidia_cc", "currently supports only"),
    ],
)
def test_unified_schema_rejects_unknown_or_unsupported_azure_values(tmp_path, field, value, message):
    config = {
        "schema_version": 1,
        "cc_deployment_mode": "azure_cc",
        "cpu_tee": "amd_sev_snp",
        "gpu_tee": "none",
        "attestation": {"service": "azure_maa"},
        "workload": {"source": {"type": "external"}},
        "azure_cc": {"deployment_target": "confidential_vm"},
    }
    config[field] = value
    project = _azure_project(tmp_path, participant=config)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    with pytest.raises(ValueError, match=message):
        CCBuilder().initialize(project, ctx)


def test_legacy_participant_fields_are_rejected(tmp_path):
    config = {
        "schema_version": 1,
        "cc_deployment_mode": "azure_cc",
        "cpu_tee": "amd_sev_snp",
        "gpu_tee": "none",
        "compute_env": "azure_cvm",
        "attestation": {"service": "azure_maa"},
        "workload": {"source": {"type": "external"}},
        "azure_cc": {"deployment_target": "confidential_vm"},
    }
    project = _azure_project(tmp_path, participant=config)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    with pytest.raises(ValueError, match="compute_env"):
        CCBuilder().initialize(project, ctx)


def test_cc_project_is_required_and_common_packager_is_mandatory(tmp_path):
    project = _azure_project(tmp_path)
    project.set_prop(PropKey.CC_PROJECT_CONFIG, None)
    with pytest.raises(ValueError, match="cc_project_config"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "one"), project))

    project = _azure_project(tmp_path)
    project.set_prop("packager", {})
    with pytest.raises(ValueError, match="common packager"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "two"), project))


def test_shared_trustee_policy_is_validated_and_applied_to_coco(tmp_path):
    project = _coco_project(tmp_path)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    builder = CCBuilder()

    builder.initialize(project, ctx)

    plan = builder.plans["site-1"]
    args = builder.deployments[plan.mode].authorizer(plan, issuer=False)["args"]
    assert args["proof_iat_leeway_seconds"] == 90
    assert args["workload_constraints"] == {"site-1": {"cpu_tee": "snp", "init_data": "a" * 64, "gpu_required": True}}
    with pytest.raises(TypeError):
        builder.plans["site-1"].internal["workload_constraints"]["site-1"]["gpu_required"] = False

    config = yaml.safe_load((tmp_path / "cc_project.yml").read_text())
    config["attestation_services"]["trustee"]["workload_constraints"]["site-1"]["init_data"] = "not-a-digest"
    _write(tmp_path / "cc_project.yml", config)
    with pytest.raises(ValueError, match="Invalid workload init_data pin"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "invalid"), project))

    config["attestation_services"]["trustee"]["workload_constraints"]["site-1"] = {
        "cpu_tee": "snp",
        "gpu_required": False,
    }
    _write(tmp_path / "cc_project.yml", config)
    with pytest.raises(ValueError, match="derived from each participant's gpu_tee"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "duplicated-gpu-policy"), project))


def test_cc_builder_reserves_logical_server_identity(tmp_path):
    project = _azure_project(tmp_path)
    project.add_client("server", "example", {})

    with pytest.raises(ValueError, match="reserve the client name 'server'"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "workspace"), project))


def test_empty_class_allow_list_adds_no_classes(tmp_path):
    config = {
        "schema_version": 1,
        "cc_deployment_mode": "azure_cc",
        "cpu_tee": "amd_sev_snp",
        "gpu_tee": "none",
        "attestation": {"service": "azure_maa"},
        "class_allow_list": [],
        "workload": {"source": {"type": "external"}},
        "azure_cc": {"deployment_target": "confidential_container"},
    }
    project = _azure_project(tmp_path, participant=config)
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    resources = _resources(ctx, project.get_server())
    builder = CCBuilder()
    builder.initialize(project, ctx)
    builder.build(project, ctx)
    assert json.loads(resources.read_text())["class_allow_list"] == []


def test_all_supplied_project_mode_blocks_are_validated(tmp_path):
    project_config = {
        "schema_version": 1,
        "attestation_services": {
            "azure_maa": {
                "type": "azure_maa",
                "endpoint": "https://sharedeus2.eus2.attest.azure.net",
                "token_expiration_seconds": 100,
                "check_frequency_seconds": 60,
            }
        },
        "build_tools": {"coco": {"build_command": "missing"}},
    }
    project = _azure_project(tmp_path, project_config=project_config)
    with pytest.raises(ValueError, match="build_tools.coco.build_command"):
        CCBuilder().initialize(project, ProvisionContext(str(tmp_path / "workspace"), project))


def test_bare_metal_and_coco_share_one_trustee_service(tmp_path):
    # This isolates dispatch validation; VaultAdapter itself has dedicated
    # image/archive integration tests.
    with patch("nvflare.lighter.cc_provision.impl.bare_metal_cvm.BareMetalCVMDeployment.bind"):
        builder_dir = tmp_path / "builder"
        builder_dir.mkdir()
        (builder_dir / "cvmctl").write_text("#!/bin/sh\n")
        (builder_dir / "cvmctl").chmod(0o700)
        for name in ("ca.pem", "admin.jwt", "approval.pub", "registry-ca.pem", "username", "password"):
            (tmp_path / name).write_text("fixture")
        public = ec.generate_private_key(ec.SECP256R1()).public_key()
        (tmp_path / "as.pem").write_bytes(
            public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        )
        (tmp_path / "archive.tar").write_text("fixture")
        context = tmp_path / "site"
        context.mkdir()
        (context / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / "platform.env").write_text("# fixture\n")
        command = tmp_path / "build"
        command.write_text("#!/bin/sh\n")
        command.chmod(0o700)
        project_config = {
            "schema_version": 1,
            "attestation_services": {
                "trustee": {
                    "type": "trustee",
                    "kbs_endpoint": "https://trustee.example.com:8443",
                    "ca_cert_file": "ca.pem",
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
            "approval": {"public_key_files": ["approval.pub"]},
            "build_tools": {
                "bare_metal_cvm": {"cvm_builder_dir": "builder"},
                "coco": {"build_command": "build"},
            },
        }
        _write(tmp_path / "cc_project.yml", project_config)
        common = {
            "schema_version": 1,
            "cpu_tee": "amd_sev_snp",
            "gpu_tee": "none",
            "attestation": {"service": "trustee"},
        }
        _write(
            tmp_path / "cc_server.yml",
            {
                **common,
                "cc_deployment_mode": "bare_metal_cvm",
                "workload": {"source": {"type": "docker_archive", "path": "archive.tar"}},
                "bare_metal_cvm": {
                    "cvm_image": "oci://registry.example.com/cvm@sha256:" + "a" * 64,
                    "storage": {},
                    "network": {},
                },
            },
        )
        _write(
            tmp_path / "cc_client.yml",
            {
                **common,
                "cc_deployment_mode": "coco",
                "workload": {"source": {"type": "docker_build", "context": "site", "dockerfile": "Dockerfile"}},
                "coco": {
                    "release_name": "site-v1",
                    "registry": "workloads",
                    "registry_repository": "workloads/site",
                    "platform_config_file": "platform.env",
                },
            },
        )
        project_dict = {
            "api_version": 3,
            "name": "mixed",
            "cc_project_config": "cc_project.yml",
            "participants": [
                {"type": "server", "name": "server.example.com", "org": "example", "cc_config": "cc_server.yml"},
                {"type": "client", "name": "site-1", "org": "example", "cc_config": "cc_client.yml"},
            ],
            "packager": {"path": CC_PACKAGER_PATH},
        }
        project = prepare_project(project_dict, project_file=tmp_path / "project.yml")
        ctx = ProvisionContext(str(tmp_path / "workspace"), project)
        builder = CCBuilder()
        builder.initialize(project, ctx)
        assert {plan.mode for plan in builder.plans.values()} == {
            CCDeploymentMode.BARE_METAL_CVM,
            CCDeploymentMode.COCO,
        }
        assert {plan.attestation_service.name for plan in builder.plans.values()} == {"trustee"}


def test_config_loaders_expose_declaring_paths(tmp_path):
    project = _azure_project(tmp_path)
    project_config = load_project_config(tmp_path / "cc_project.yml")
    normalized = load_participant_config(tmp_path / "cc_server.yml", project_config)
    assert project.get_server().get_prop(PropKey.CC_CONFIG) == "cc_server.yml"
    assert project_config["_config_path"] == tmp_path / "cc_project.yml"
    assert normalized["config_path"] == tmp_path / "cc_server.yml"


def test_bare_metal_adapter_uses_content_addressed_project_config(tmp_path):
    project = prepare_project(
        {
            "api_version": 3,
            "name": "bare",
            "participants": [{"type": "server", "name": "server.example.com", "org": "example"}],
        },
        project_file=tmp_path / "project.yml",
    )
    ctx = ProvisionContext(str(tmp_path / "workspace"), project)
    for name in ("ca.pem", "admin.jwt", "approval.pub"):
        (tmp_path / name).write_text("fixture")
    project_config = {
        "_config_path": tmp_path / "cc_project.yml",
        "approval": {"public_key_files": ["approval.pub"]},
        "build_tools": {"bare_metal_cvm": {"cvm_builder_dir": "builder"}},
    }
    mode = {
        "cvm_image": "oci://registry.example/cvm@sha256:" + "a" * 64,
        "storage": {
            "vault_size_gib": 8,
            "applog_size_gib": 1,
            "user_config_size_gib": 1,
            "user_data_size_gib": 1,
        },
        "network": {
            "allowed_in_ports": [],
            "allowed_out_ports": [],
            "allowed_in_cidrs": [],
            "allowed_out_cidrs": [],
        },
    }
    deployment = BareMetalCVMDeployment()
    paths = []
    with patch("nvflare.lighter.cc_provision.impl.bare_metal_cvm.VaultAdapter") as adapter:
        for endpoint in ("https://trustee-one.example:8443", "https://trustee-two.example:8443"):
            plan = SimpleNamespace(
                participant_name="server.example.com",
                config_path=tmp_path / "cc_server.yml",
                attestation_service=SimpleNamespace(
                    config_path=tmp_path / "cc_project.yml",
                    values={
                        "kbs_endpoint": endpoint,
                        "ca_cert_file": "ca.pem",
                        "admin_token_file": "admin.jwt",
                    },
                ),
                mode_config=mode,
                workload_source=SimpleNamespace(values={"path": tmp_path / "application.tar"}),
                cpu_tee=CPUTEE.AMD_SEV_SNP,
                gpu_tee=GPUTEE.NONE,
            )
            deployment.bind(plan, project_config, project, ctx)
            paths.append(Path(adapter.call_args.args[0]["project_config"]))

    assert paths[0] != paths[1]
    assert "trustee-one.example" in paths[0].read_text()
    assert "trustee-two.example" in paths[1].read_text()
    assert all(path.stat().st_mode & 0o077 == 0 for path in paths)


def test_bare_metal_deployment_returns_common_artifact_result(tmp_path):
    deployment = BareMetalCVMDeployment()
    external = tmp_path / "external-output"
    external.mkdir()
    archive = external / "delivery.oci.tar"
    archive.write_bytes(b"reviewed CVM delivery")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    adapter = SimpleNamespace(
        build=lambda ctx, source_dirs: [
            {
                "artifacts": [
                    {
                        "path": str(archive),
                        "archive_sha256": archive_sha256,
                        "platform": "intel_tdx",
                        "manifest_digest": "sha256:" + "b" * 64,
                        "cvm_build_id": "reviewed-build",
                        "resource": "keys/reviewed-build/vault-key",
                    }
                ]
            }
        ]
    )
    deployment.adapters["server.example.com"] = adapter
    plan = SimpleNamespace(
        participant_name="server.example.com",
        mode=CCDeploymentMode.BARE_METAL_CVM,
        cpu_tee=CPUTEE.INTEL_TDX,
        gpu_tee=GPUTEE.NONE,
        attestation_service=SimpleNamespace(name="trustee"),
    )

    result = deployment.package(plan, tmp_path / "private-kit", tmp_path / "public", {})

    assert result.mode is CCDeploymentMode.BARE_METAL_CVM
    assert result.artifacts[0].path == "delivery.oci.tar"
    assert (tmp_path / "public/delivery.oci.tar").read_bytes() == archive.read_bytes()
    assert result.artifacts[0].sha256 == archive_sha256
    assert result.artifacts[0].metadata["manifest_digest"] == "sha256:" + "b" * 64


def test_azure_end_to_end_provisioning_emits_common_manifest(tmp_path):
    project = _azure_project(tmp_path)
    provisioner = Provisioner(
        str(tmp_path / "output"),
        [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(), CCBuilder(), SignatureBuilder()],
        CCPackager(),
    )

    ctx = provisioner.provision(project)

    assert ctx[CtxKey.PROVISION_SUCCESS] is True
    output = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    kit = output / "server.example.com"
    assert verify_folder_signature(
        str(kit), str(kit / "startup/rootCA.pem"), single_signer=True, signature_file=ProvFileName.SIGNATURE_JSON
    )
    manifest = json.loads((output / "cc_manifests/server.example.com.json").read_text())
    assert manifest["schema"] == "nvflare-cc-delivery/v1"
    assert manifest["cc_deployment_mode"] == "azure_cc"
    assert manifest["artifacts"][0]["type"] == "azure_startup_kit"
    assert not (output / ProvFileName.START_ALL_SH).exists()
    assert (Path(ctx.get_state_dir()) / "cc-private" / output.name / "server.example.com/startup-kit").is_dir()


def test_common_packager_retains_private_stage_when_prod_number_is_reused(tmp_path):
    project = _azure_project(tmp_path)
    root = tmp_path / "output/unified"
    private = root / "state/cc-private/prod_00"
    retained = []
    for attempt in range(3):
        if attempt:
            shutil.rmtree(root / "prod_00")
        provisioner = Provisioner(
            str(tmp_path / "output"),
            [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(), CCBuilder(), SignatureBuilder()],
            CCPackager(),
        )
        ctx = provisioner.provision(project)
        assert ctx[CtxKey.PROVISION_SUCCESS] is True
        archives = sorted(private.parent.glob("prod_00.superseded-*"))
        assert len(archives) == attempt
        for archive in archives:
            previous = archive / "prod_00"
            index = int((previous / "retained-marker.txt").read_text())
            assert {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()} == retained[
                index
            ]
        (private / "retained-marker.txt").write_text(str(attempt))
        retained.append({p.relative_to(private): p.read_bytes() for p in private.rglob("*") if p.is_file()})


def test_coco_end_to_end_provisioning_uses_declared_source_registry_and_common_manifest(tmp_path):
    from tests.unit_test.lighter.cc_provision.impl.test_coco import coco_pod

    project = _coco_project(tmp_path)

    def runner(command, **kwargs):
        request = json.loads(Path(command[1]).read_text())
        workload = dict(
            shlex.split(line)[0].split("=", 1) for line in Path(request["workload_env"]).read_text().splitlines()
        )
        assert workload["REGISTRY_ENDPOINT"] == "registry.example.com:5000"
        assert workload["KBS_URL"] == "https://trustee.example.com:8443"
        assert Path(workload["BUILD_CONTEXT"]).joinpath(".nvflare-kit/signature.json").is_file()
        image = "registry.example.com:5000/workloads/site@sha256:" + "a" * 64
        pod = Path(request["result_file"]).with_name("pod.yaml")
        pod.write_text(yaml.safe_dump(coco_pod("kata-qemu-nvidia-gpu-snp", "nvidia", image)))
        Path(request["result_file"]).write_text(
            json.dumps(
                {
                    "schema": "nvflare-coco-build-result/v1",
                    "release_name": "site-v1",
                    "pod_yaml": str(pod),
                }
            )
        )

    provisioner = Provisioner(
        str(tmp_path / "output"),
        [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(), CCBuilder(), SignatureBuilder()],
        CCPackager(),
    )
    with patch("nvflare.lighter.cc_provision.impl.cc_packager.subprocess.run", side_effect=runner):
        ctx = provisioner.provision(project)

    assert ctx[CtxKey.PROVISION_SUCCESS] is True, ctx.get_errors()
    output = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    assert [path.name for path in (output / "site-1").iterdir()] == ["site-v1-pod.yaml"]
    manifest = json.loads((output / "cc_manifests/site-1.json").read_text())
    assert manifest["cc_deployment_mode"] == "coco"
    assert manifest["artifacts"][0]["image"].endswith("@sha256:" + "a" * 64)
    assert (Path(ctx.get_state_dir()) / "cc-private" / output.name / "site-1/startup-kit").is_dir()
