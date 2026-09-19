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
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nvflare.lighter.cc_provision.impl.cc import CCBuilder
from nvflare.lighter.cc_provision.impl.coco import CoCoBuilder, resolve_cc_config, validate_coco_config
from nvflare.lighter.cc_provision.impl.coco_packager import COMMAND, CoCoPackager
from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import verify_folder_signature


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


def test_relative_cc_config_requires_explicit_source(tmp_path, monkeypatch):
    config = {"api_version": 3, "name": "paths", "participants": []}
    project = prepare_project(config.copy())
    with pytest.raises(ValueError, match="project_file"):
        resolve_cc_config(project, "cc_site.yml")
    source = tmp_path / "source/project.yaml"
    project = prepare_project(config, project_file=source)
    monkeypatch.chdir(tmp_path)
    assert resolve_cc_config(project, "cc_site.yml") == str(source.parent / "cc_site.yml")


def write_fake_result(request, config=None):
    pytest.importorskip("tomllib", reason="full CoCo Pod packaging requires a Python 3.11+ deployment host")
    from tests.unit_test.lighter.cc_provision.impl.workload_security_context_test import context, policy, policy_data

    owner = request.parent
    params = json.loads(request.read_text())
    pod_path = owner / "protected-pod.yaml"
    config = config or {"registry_repository": "workloads/site-1", "release_name": "site-1-v1"}
    image = "secure.unit.local:5000/" + config["registry_repository"] + "@sha256:" + "a" * 64
    data = policy_data()
    data["containers"][0]["OCI"]["Annotations"]["io.kubernetes.cri.image-name"] = image
    data["containers"][0]["OCI"]["Process"]["Args"] = COMMAND
    initdata = '[data]\n"policy.rego" = ' + "'''\n" + policy(data) + "\n'''\n"
    pod_path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "annotations": {
                        "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                            gzip.compress(initdata.encode())
                        ).decode()
                    }
                },
                "spec": {
                    "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
                    "automountServiceAccountToken": False,
                    "enableServiceLinks": False,
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "image": image,
                            "command": COMMAND,
                            "imagePullPolicy": "Always",
                            "securityContext": context(),
                            "stdin": False,
                            "tty": False,
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
                "release_name": config["release_name"],
                "pod_yaml": str(pod_path),
            }
        )
    )


def setup_server_project(tmp_path, with_cc_client=True):
    project, client_config = setup_project(tmp_path)
    server_config = copy.deepcopy(client_config)
    server_config.update(
        role="server",
        release_name="server-v1",
        registry_repository="workloads/server",
        class_allow_list=["my_app.controller.ReviewedController"],
    )
    (tmp_path / "cc_server.yml").write_text(yaml.safe_dump(server_config))
    project.get_server().set_prop(PropKey.CC_CONFIG, "cc_server.yml")
    configs = {project.get_server().name: server_config}
    if with_cc_client:
        configs["site-1"] = client_config
    else:
        project.get_clients()[0].set_prop(PropKey.CC_CONFIG, None)
    return project, configs


@pytest.mark.parametrize("with_cc_client", [False, True])
def test_provision_server_signed_kit_and_client_verifiers(tmp_path, with_cc_client):
    project, configs = setup_server_project(tmp_path, with_cc_client)
    seen = {}
    root = tmp_path / "workspace/test_project"

    def runner(command, **kwargs):
        request = Path(command[1])
        owner = request.parent
        config = configs[owner.name]
        if not seen:
            for name in configs:
                assert not (root / "prod_00" / name).exists()
        kit = owner / "build-context/.nvflare-kit"
        assert (kit / "signature.json").is_file()
        assert (kit / "startup" / f'{config["role"]}.key').is_file()
        if config["role"] == "server":
            assert not (kit / "startup/client.key").exists()
        seen[owner.name] = owner
        write_fake_result(request, config)

    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run", side_effect=runner):
        ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get_errors()
    assert set(seen) == set(configs)
    result = Path(ctx.get_result_location())
    expected_sites = {"server", "site-1"} if with_cc_client else {"server"}
    expected_verifiers = {site: ["coco_authorizer"] for site in expected_sites}
    for participant in project.get_all_participants():
        protected = participant.name in configs
        assert bool(participant.get_prop(PropKey.CC_ENABLED)) == protected
        kit = seen[participant.name] / "startup-kit" if protected else result / participant.name
        local = kit / "local"
        manager = json.loads((local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = json.loads((local / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
        assert set(manager["cc_enabled_sites"]) == expected_sites
        assert manager["required_site_verifier_ids"] == expected_verifiers
        assert manager["cc_verifier_ids"] == ["coco_authorizer"]
        assert manager["require_site_binding"] is True
        if protected:
            assert verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            )
            logical_site = "server" if participant.type == "server" else participant.name
            assert authorizer["site_name"] == logical_site
            assert manager["cc_issuers_conf"] == [{"issuer_id": "coco_authorizer", "token_expiration": 300}]
            owner = seen[participant.name]
            key_name = f"startup/{participant.type}.key"
            assert (kit / key_name).read_bytes() == (owner / "build-context/.nvflare-kit" / key_name).read_bytes()
            assert owner.stat().st_mode & 0o777 == 0o700
            assert (owner / "build-request.json").stat().st_mode & 0o777 == 0o600
            assert json.dumps(COMMAND) in (owner / "build-context/Dockerfile.coco").read_text()
            handoff = result / participant.name
            assert [p.name for p in handoff.iterdir()] == [configs[participant.name]["release_name"] + "-pod.yaml"]
            pod = yaml.safe_load(next(handoff.iterdir()).read_text())
            assert pod["spec"]["containers"][0]["command"] == COMMAND
            if participant.type == "server":
                permissions = json.loads((local / ProvFileName.AUTHORIZATION_JSON_DEFAULT).read_text())["permissions"]
                assert permissions["org_admin"]["submit_job"] == "none"
                assert permissions["org_admin"]["shell_commands"] == "none"
                assert permissions["org_admin"]["byoc"] == "none"
                resources = json.loads((local / ProvFileName.RESOURCES_JSON_DEFAULT).read_text())
                assert "my_app.controller.ReviewedController" in resources["class_allow_list"]
        else:
            assert manager["cc_issuers_conf"] == []
            assert "site_name" not in authorizer
            assert (kit / "startup/client.key").is_file()


def test_server_entrypoint_verifies_kit_then_starts_server_module(tmp_path, monkeypatch):
    project, configs = setup_server_project(tmp_path, with_cc_client=False)

    def runner(command, **kwargs):
        request = Path(command[1])
        write_fake_result(request, configs[request.parent.name])

    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run", side_effect=runner):
        ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get_errors()
    kit = tmp_path / "workspace/test_project/state/coco-private/prod_00/server.example.com/build-context/.nvflare-kit"
    assert verify_folder_signature(
        str(kit), str(kit / "startup/rootCA.pem"), single_signer=True, signature_file=ProvFileName.SIGNATURE_JSON
    )
    binaries = tmp_path / "test-bin"
    binaries.mkdir()
    recorder = binaries / "python3"
    recorder.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['COCO_TEST_ARGS_FILE'], 'a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    recorder.chmod(0o700)
    recorded = tmp_path / "python-arguments.jsonl"
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("NVFL_WORKSPACE", str(kit))
    monkeypatch.setenv("COCO_TEST_ARGS_FILE", str(recorded))
    subprocess.run([str(kit / "startup/sub_start.sh"), *COMMAND[1:]], check=True, capture_output=True, timeout=10)
    verification, startup = [json.loads(line) for line in recorded.read_text().splitlines()]
    assert verification == [
        "-m",
        "nvflare.tool.verify_startup_kits",
        "-f",
        str(kit),
        "-c",
        str(kit / "startup/rootCA.pem"),
    ]
    assert startup == [
        "-u",
        "-m",
        "nvflare.private.fed.app.server.server_train",
        "-m",
        str(kit),
        "-s",
        "fed_server.json",
        "--set",
        "secure_train=true",
        "org=example",
        "config_folder=",
    ]


@pytest.mark.parametrize("custom_retry", [False, True, "legacy_timeout"])
def test_provision_real_signed_kit_then_package(tmp_path, custom_retry):
    project, config = setup_project(tmp_path)
    retry_options = {}
    timeouts = {"registration_token_timeout": 300, "refresh_token_timeout": 22.5, "get_token_request_timeout": 45}
    if custom_retry == "legacy_timeout":
        config["cc_attestation"]["get_token_request_timeout"] = 10
        timeouts.update(get_token_request_timeout=10, refresh_token_timeout=5)
        (tmp_path / "cc_site-1.yml").write_text(yaml.safe_dump(config))
    elif custom_retry:
        retry_options = {
            "retry_max_attempts": 6,
            "retry_initial_delay": 2,
            "retry_max_delay": 12,
            "retry_backoff_multiplier": 3,
            "retry_jitter_ratio": 0.25,
        }
        timeouts = {"registration_token_timeout": 180, "refresh_token_timeout": 40, "get_token_request_timeout": 55}
        config["cc_issuers"][0]["args"].update(retry_options)
        config["cc_attestation"].update(timeouts)
        (tmp_path / "cc_site-1.yml").write_text(yaml.safe_dump(config))
    seen = []

    def runner(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 3600
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
    assert manager["cc_verifier_ids"] == ["coco_authorizer"]
    owner = seen[0].parent
    client_local = owner / "startup-kit/local"
    client_manager = json.loads((client_local / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
    assert client_manager["cc_verifier_ids"] == ["coco_authorizer"]
    assert (
        client_manager["required_site_verifier_ids"]
        == manager["required_site_verifier_ids"]
        == {"site-1": ["coco_authorizer"]}
    )
    client_auth = json.loads((client_local / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
    assert client_auth["site_name"] == "site-1"
    for name, value in retry_options.items():
        assert client_auth[name] == value
    for name, value in timeouts.items():
        assert client_manager[name] == manager[name] == value
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("registration_token_timeout", 0),
        ("refresh_token_timeout", True),
        ("get_token_request_timeout", 0),
        ("get_token_request_timeout", float("inf")),
        ("unknown_retry_option", 1),
    ],
)
def test_invalid_provisioning_retry_timeout(tmp_path, field, value):
    _, config = setup_project(tmp_path)
    config["cc_attestation"][field] = value
    with pytest.raises(ValueError):
        validate_coco_config(config)


def test_explicit_conflicting_provisioning_timeout_is_rejected(tmp_path):
    _, config = setup_project(tmp_path)
    config["cc_attestation"].update(get_token_request_timeout=10, refresh_token_timeout=30)
    with pytest.raises(ValueError, match="must exceed"):
        validate_coco_config(config)


def test_invalid_backoff_fails_before_packaging(tmp_path):
    project, config = setup_project(tmp_path)
    config["cc_issuers"][0]["args"]["retry_jitter_ratio"] = 2
    (tmp_path / "cc_site-1.yml").write_text(yaml.safe_dump(config))
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run") as runner:
        ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)
    runner.assert_not_called()
    assert not list((tmp_path / "workspace/test_project").glob("prod_*"))


@pytest.mark.parametrize("different_timeout", [False, True])
def test_clients_may_vary_backoff_but_must_share_manager_timeouts(tmp_path, different_timeout):
    project, config = setup_project(tmp_path)
    first, second = project.get_clients()
    first.set_prop(PropKey.CC_CONFIG_DICT, config)
    other = copy.deepcopy(config)
    other["release_name"] = "site-2-v1"
    other["cc_issuers"][0]["args"]["retry_initial_delay"] = 2
    other["cc_attestation"]["get_token_request_timeout"] = 60 if different_timeout else 45
    second.set_prop(PropKey.CC_CONFIG_DICT, other)
    second.set_prop(PropKey.CC_CONFIG, "cc_site-2.yml")
    (tmp_path / "cc_site-2.yml").write_text(yaml.safe_dump(other))
    builder = CoCoBuilder()
    if different_timeout:
        with pytest.raises(ValueError, match="attestation timing"):
            builder.initialize(project, None)
    else:
        builder.initialize(project, None)
        assert builder.settings[second.name][0]["retry_initial_delay"] == 2


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


@pytest.mark.parametrize("failure_stage", ["prepare", "build"])
def test_server_and_client_kits_are_private_before_first_build_failure(tmp_path, failure_stage):
    project, configs = setup_server_project(tmp_path)
    root = tmp_path / "workspace/test_project"
    if failure_stage == "prepare":
        configs[project.get_server().name]["image_build"]["context"] = "missing-context"
        (tmp_path / "cc_server.yml").write_text(yaml.safe_dump(configs[project.get_server().name]))

    def runner(command, **kwargs):
        for name, config in configs.items():
            assert not (root / "prod_00" / name).exists()
            kit = root / "state/coco-private/prod_00" / name / "startup-kit"
            assert (kit / "startup" / f'{config["role"]}.key').is_file()
        raise subprocess.CalledProcessError(1, command)

    error_type = ValueError if failure_stage == "prepare" else subprocess.CalledProcessError
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run", side_effect=runner) as build:
        with pytest.raises(error_type):
            Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    if failure_stage == "prepare":
        build.assert_not_called()
    else:
        assert build.call_count == 1
    for name, config in configs.items():
        assert not (root / "prod_00" / name).exists()
        private = root / "state/coco-private/prod_00" / name / "startup-kit"
        assert (private / "startup" / f'{config["role"]}.key').is_file()


@pytest.mark.parametrize("timeout", [None, True, False, 0, -1, 1.5, "60"])
def test_invalid_build_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="build_timeout"):
        CoCoPackager(build_image_cmd="reviewed-builder", build_timeout=timeout)


def test_build_timeout_preserves_private_kit_without_public_handoff(tmp_path):
    project, _ = setup_project(tmp_path)
    (tmp_path / "build.sh").write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    with pytest.raises(subprocess.TimeoutExpired) as error:
        Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh", build_timeout=1)).provision(
            project
        )
    # subprocess may report the remaining deadline after process startup.
    assert 0 < error.value.timeout <= 1
    root = tmp_path / "workspace/test_project"
    assert not (root / "prod_00/site-1").exists()
    assert (root / "state/coco-private/prod_00/site-1/startup-kit/startup/client.key").is_file()


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_reused_private_stage_must_be_a_regular_directory(tmp_path, kind):
    project, _ = setup_project(tmp_path)
    private = tmp_path / "workspace/test_project/state/coco-private/prod_00"
    private.parent.mkdir(parents=True)
    if kind == "file":
        private.write_text("retain this file")
    else:
        target = tmp_path / "previous-stage"
        if kind == "symlink":
            target.mkdir()
        private.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular private stage directory"):
        Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    if kind == "file":
        assert private.read_text() == "retain this file"
    else:
        assert private.is_symlink()


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


@pytest.mark.parametrize("participant_type", ["client", "server"])
def test_config_role_must_match_participant_type(tmp_path, participant_type):
    project, configs = setup_server_project(tmp_path)
    participant = project.get_server() if participant_type == "server" else project.get_clients()[0]
    config = configs[participant.name]
    config["role"] = "client" if participant_type == "server" else "server"
    (tmp_path / participant.get_prop(PropKey.CC_CONFIG)).write_text(yaml.safe_dump(config))
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run") as runner:
        ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)
    assert "role" in " ".join(ctx.get_errors()).lower()
    runner.assert_not_called()
    assert not list((tmp_path / "workspace/test_project").glob("prod_*"))


@pytest.mark.parametrize("participant_type", ["client", "server"])
def test_packager_revalidates_participant_role(tmp_path, participant_type):
    project, configs = setup_server_project(tmp_path)
    ctx = Provisioner(str(tmp_path / "workspace"), builders()).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR), ctx.get_errors()
    participant = project.get_server() if participant_type == "server" else project.get_clients()[0]
    config = configs[participant.name]
    config["role"] = "client" if participant_type == "server" else "server"
    participant.set_prop(PropKey.CC_CONFIG_DICT, config)
    (tmp_path / participant.get_prop(PropKey.CC_CONFIG)).write_text(yaml.safe_dump(config))
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run") as runner:
        with pytest.raises(ValueError, match="role must match participant type"):
            CoCoPackager("build.sh").package(project, ctx)
    runner.assert_not_called()


@pytest.mark.parametrize("role", [None, "", "admin", "relay", True])
def test_unsupported_coco_roles_are_rejected(tmp_path, role):
    _, config = setup_project(tmp_path)
    config["role"] = role
    with pytest.raises(ValueError, match="role"):
        validate_coco_config(config)


def test_client_name_cannot_shadow_server_runtime_identity(tmp_path):
    project, _ = setup_server_project(tmp_path)
    project.add_client("server", "example", {})
    with patch("nvflare.lighter.cc_provision.impl.coco_packager.subprocess.run") as runner:
        ctx = Provisioner(str(tmp_path / "workspace"), builders(), CoCoPackager("build.sh")).provision(project)
    assert ctx.get(CtxKey.BUILD_ERROR)
    assert "reserved server runtime identity" in " ".join(ctx.get_errors())
    runner.assert_not_called()
    assert not list((tmp_path / "workspace/test_project").glob("prod_*"))


@pytest.mark.parametrize("first_build_fails", [False, True])
def test_cli_reprovision_retains_private_stages_from_other_directory(tmp_path, first_build_fails):
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
    fixture_request = tmp_path / "fixture-request.json"
    fixture_request.write_text(json.dumps({"result_file": str(tmp_path / "fixture-result.json")}))
    write_fake_result(fixture_request)
    pod = yaml.safe_load((tmp_path / "protected-pod.yaml").read_text())
    (tmp_path / "build.sh").write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "out = Path(request['result_file'])\npod = out.parent / 'pod.yaml'\n"
        f"pod.write_text({json.dumps(json.dumps(pod))})\n"
        "out.write_text(json.dumps({'schema': 'nvflare-coco-build-result/v1', 'release_name': 'site-1-v1', 'pod_yaml': str(pod)}))\n"
    )
    command = [
        str(Path(sys.executable).parent / "nvflare"),
        "provision",
        "-p",
        str(tmp_path / "project.yaml"),
        "-w",
        str(tmp_path / "workspace"),
        "--force",
    ]
    good_runner = (tmp_path / "build.sh").read_text()
    if first_build_fails:
        (tmp_path / "build.sh").write_text("#!/bin/sh\nexit 99\n")
    root = tmp_path / "workspace/test_project"
    private = root / "state/coco-private/prod_00"
    retained = []
    for attempt in range(3):
        if attempt:
            # Only remove the generated prod directory in this test's private
            # temporary workspace, matching the documented stage-reuse path.
            shutil.rmtree(root / "prod_00")
            (tmp_path / "build.sh").write_text(good_runner)
        result = subprocess.run(command, cwd=tmp_path.parent, text=True, capture_output=True, timeout=30)
        public = root / "prod_00/site-1"
        if attempt == 0 and first_build_fails:
            assert result.returncode != 0
            assert not public.exists()
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert sorted(p.name for p in public.iterdir()) == ["site-1-v1-pod.yaml"]
        assert (private / "site-1/startup-kit/startup/client.key").is_file()
        assert not (private / "retained-marker.txt").exists()
        archives = sorted(private.parent.glob("prod_00.superseded-*"))
        assert len(archives) == attempt
        for archive in archives:
            assert archive.stat().st_mode & 0o777 == 0o700
            previous = archive / "prod_00"
            index = int((previous / "retained-marker.txt").read_text())
            assert {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()} == retained[
                index
            ]
        (private / "retained-marker.txt").write_text(str(attempt))
        retained.append({p.relative_to(private): p.read_bytes() for p in private.rglob("*") if p.is_file()})
