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

import base64
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.impl.cc import CCBuilder
from nvflare.lighter.cc_provision.impl.coco import validate_coco_config
from nvflare.lighter.cc_provision.impl.coco_packager import COMMAND, CoCoPackager
from nvflare.lighter.constants import CtxKey, PropKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner


def setup_project(tmp_path):
    config = {
        "compute_env": "confidential_containers",
        "cc_cpu_mechanism": "amd_sev_snp",
        "cc_gpu": "nvidia",
        "role": "client",
        "image_build": {"context": "site-1", "dockerfile": "Dockerfile"},
        "release_name": "site-1-v1",
        "registry_repository": "workloads/site-1",
        "platform_config": "admin/platform.env",
        "cc_issuers": [
            {
                "id": "coco_authorizer",
                "path": "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer",
                "token_expiration": 300,
                "args": {"trustee_public_key_file": "trustee-as-public.pem"},
            }
        ],
        "cc_attestation": {"check_frequency": 120},
    }
    public = ec.generate_private_key(ec.SECP256R1()).public_key()
    (tmp_path / "trustee-as-public.pem").write_bytes(
        public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    (tmp_path / "cc_site-1.yml").write_text(yaml.safe_dump(config))
    context = tmp_path / "site-1"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM reviewed-base\n")
    (context / ".dockerignore").write_text(".*\n")
    admin = tmp_path / "admin"
    admin.mkdir()
    (admin / "platform.env").write_text("# fixture\n")
    runner = tmp_path / "build.sh"
    runner.write_text("#!/bin/sh\nexit 99\n")
    runner.chmod(0o700)
    project = prepare_project(
        {
            "api_version": 3,
            "name": "test_project",
            "participants": [
                {"type": "server", "name": "server.example.com", "org": "example"},
                {"type": "client", "name": "site-1", "org": "example", "cc_config": "cc_site-1.yml"},
                {"type": "client", "name": "plain-client", "org": "example"},
            ],
            "packager": {"path": "nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager"},
        }
    )
    project.set_prop("_project_file", str(tmp_path / "project.yaml"))
    return project, config


def builders():
    return [WorkspaceBuilder(), StaticFileBuilder(), CertBuilder(), CCBuilder(), SignatureBuilder()]


def write_fake_result(request):
    owner = request.parent
    params = json.loads(request.read_text())
    pod_path = owner / "protected-pod.yaml"
    pod_path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "annotations": {
                        "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                            gzip.compress(b"fixture")
                        ).decode()
                    }
                },
                "spec": {
                    "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
                    "containers": [
                        {
                            "image": "secure.unit.local:5000/workloads/site-1@sha256:" + "a" * 64,
                            "command": COMMAND,
                            "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                        }
                    ],
                },
            }
        )
    )
    Path(params["result_file"]).write_text(
        json.dumps(
            {
                "schema": "nvflare-coco-build-result/v1",
                "release_name": "site-1-v1",
                "pod_yaml": str(pod_path),
            }
        )
    )


def test_provision_real_signed_kit_then_package(tmp_path):
    project, _ = setup_project(tmp_path)
    seen = []

    def runner(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] == tmp_path
        request = Path(command[1])
        kit = request.parent / "build-context/.nvflare-kit"
        assert (kit / "signature.json").is_file()
        assert (kit / "startup/client.key").is_file()
        assert not (tmp_path / "workspace/test_project/prod_00/site-1").exists()
        seen.append(request)
        write_fake_result(request)

    provisioner = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh"))
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run", side_effect=runner):
        ctx = provisioner.provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    assert len(seen) == 1
    result = Path(ctx.get_result_location())
    assert [p.name for p in (result / "site-1").iterdir()] == ["site-1-v1-pod.yaml"]
    assert (result / "plain-client/startup/client.key").is_file()
    assert (result / "server.example.com/startup/server.key").is_file()
    assert not project.get_server().get_prop(PropKey.CC_ENABLED)
    local = result / "server.example.com/local"
    manager = json.loads((local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
    assert manager["cc_issuers_conf"] == []
    assert manager["required_namespaces"] == ["x-trustee-coco"]
    verifier = json.loads((local / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
    assert verifier["expected_workloads"]["site-1"]["init_data"] == hashlib.sha256(b"fixture").hexdigest()
    owner = seen[0].parent
    client_local = owner / "startup-kit/local"
    client_manager = json.loads((client_local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
    assert client_manager["verify_peer_tokens"] is False
    assert client_manager["cc_verifier_ids"] == []
    client_auth = json.loads((client_local / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
    assert client_auth["site_name"] == "site-1"
    assert "expected_workloads" not in client_auth
    assert owner.stat().st_mode & 0o777 == 0o700
    assert (owner / "build-request.json").stat().st_mode & 0o777 == 0o600
    assert (owner / "startup-kit/startup/client.key").read_bytes() == (
        owner / "build-context/.nvflare-kit/startup/client.key"
    ).read_bytes()
    dockerfile = (owner / "build-context/Dockerfile.coco").read_text()
    assert "COPY --chown=65532:65532 .nvflare-kit/ /opt/nvflare/" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert json.dumps(COMMAND) in dockerfile
    assert "!.nvflare-kit/**" in (owner / "build-context/.dockerignore").read_text()
    assert "APP_READ_ONLY_ROOT_FILESYSTEM=false" in (owner / "workload.env").read_text()


@pytest.mark.parametrize(
    "field,value",
    [("cc_gpu", "none"), ("cc_cpu_mechanism", "intel_tdx"), ("role", "server"), ("release_name", "../oops")],
)
def test_invalid_config_is_fail_closed(tmp_path, field, value):
    project, config = setup_project(tmp_path)
    config[field] = value
    (tmp_path / "cc_site-1.yml").write_text(yaml.safe_dump(config))
    ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)
    assert not list((tmp_path / "workspace/test_project").glob("prod_*"))


def test_missing_gpu_and_unknown_fields_rejected(tmp_path):
    _, config = setup_project(tmp_path)
    config.pop("cc_gpu")
    with pytest.raises(ValueError, match="cc_gpu"):
        validate_coco_config(config)
    config["cc_gpu"] = "nvidia"
    config["unknown_field"] = []
    with pytest.raises(ValueError, match="fields"):
        validate_coco_config(config)


def test_missing_packager_rejected_before_plaintext_release(tmp_path):
    project, _ = setup_project(tmp_path)
    project.set_prop("packager", {})
    ctx = Provisioner(str(tmp_path / "workspace"), builders()).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)


def test_build_failure_preserves_private_kit_without_public_handoff(tmp_path):
    project, _ = setup_project(tmp_path)
    with patch(
        "nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["build.sh"]),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    root = tmp_path / "workspace/test_project"
    assert not (root / "prod_00/site-1").exists()
    assert (root / "state/coco-private/prod_00/site-1/startup-kit/startup/client.key").is_file()


def test_context_symlink_rejected(tmp_path):
    project, _ = setup_project(tmp_path)
    (tmp_path / "site-1/leak").symlink_to(tmp_path / "admin/platform.env")
    with pytest.raises(ValueError, match="regular files"):
        Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)


def test_non_coco_regression_and_missing_config_fail_closed(tmp_path):
    project, _ = setup_project(tmp_path)
    (tmp_path / "cc_site-1.yml").unlink()
    ctx = Provisioner(str(tmp_path / "workspace"), builders()).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)


def test_coco_on_server_rejected(tmp_path):
    project, _ = setup_project(tmp_path)
    project.get_server().set_prop(PropKey.CC_CONFIG, "cc_site-1.yml")
    ctx = Provisioner(str(tmp_path / "workspace"), builders()).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)


def test_cli_resolves_project_relative_cc_config_from_other_directory(tmp_path):
    project, _ = setup_project(tmp_path)
    definition = {
        "api_version": 3,
        "name": "test_project",
        "participants": [
            {"name": "server.example.com", "type": "server", "org": "example"},
            {"name": "site-1", "type": "client", "org": "example", "cc_config": "cc_site-1.yml"},
        ],
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.cc_provision.impl.cc.CCBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
        "packager": {
            "path": "nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager",
            "args": {"build_image_cmd": "build.sh"},
        },
    }
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(definition))
    # Trusted fixture runner: publish no image and generate only a fake receipt.
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "annotations": {
                "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(gzip.compress(b"fixture")).decode()
            }
        },
        "spec": {
            "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
            "containers": [
                {
                    "command": COMMAND,
                    "image": "secure.unit.local:5000/workloads/site-1@sha256:" + "a" * 64,
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                }
            ],
        },
    }
    (tmp_path / "build.sh").write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "out = Path(request['result_file'])\npod = out.parent / 'pod.yaml'\n"
        f"pod.write_text({json.dumps(json.dumps(pod))})\n"
        "out.write_text(json.dumps({'schema': 'nvflare-coco-build-result/v1', 'release_name': 'site-1-v1', 'pod_yaml': str(pod)}))\n"
    )
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "nvflare"),
            "provision",
            "-p",
            str(tmp_path / "project.yaml"),
            "-w",
            str(tmp_path / "workspace"),
            "--force",
        ],
        cwd=tmp_path.parent,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    public = tmp_path / "workspace/test_project/prod_00/site-1"
    assert sorted(p.name for p in public.iterdir()) == ["site-1-v1-pod.yaml"]
