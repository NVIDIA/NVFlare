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

import copy
import hashlib
import io
import json
import re
import shutil
import stat
import tarfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nvflare.lighter.cc import vault_adapter as adapter_module
from nvflare.lighter.cc.vault_adapter import VaultAdapter, docker_image_id, invoke_vault_builder
from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project, provision
from nvflare.lighter.spec import Builder
from nvflare.lighter.utils import verify_folder_signature

REGISTRY_IMAGE = "registry.example.org/cvm/cpu@sha256:" + "d" * 64


def write_json(path, data):
    path.write_text(json.dumps(data))


def write_tar(path, members, mode="w"):
    with tarfile.open(path, mode) as archive:
        for name, raw in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def write_docker_archive(path, *, modern=False, extra=None, mode="w", os_name="linux", architecture="amd64"):
    config = json.dumps({"os": os_name, "architecture": architecture, "config": {"Env": ["TEST=1"]}}).encode()
    digest = hashlib.sha256(config).hexdigest()
    name = "blobs/sha256/" + digest if modern else digest + ".json"
    entries = [{"Config": name, "RepoTags": ["test:latest"], "Layers": []}]
    members = {name: config}
    if extra:
        entries.append({"Config": "extra.json", "Layers": []})
        members["extra.json"] = extra
    members["manifest.json"] = json.dumps(entries).encode()
    write_tar(path, members, mode)
    return "sha256:" + digest


def write_profile(directory, gpu="none"):
    contract = {"gpu": gpu, "bootstrap_egress": [443, 8443]}
    bundles = {}
    for platform in ("amd_sev_snp", "intel_tdx"):
        bundle = directory / platform
        bundle.mkdir(exist_ok=True)
        manifest = {
            "build_id": "generic-" + platform,
            "platform": platform,
            "profile_version": "cpu-2026.09",
            "contract": contract,
        }
        write_json(bundle / "cvm_manifest.json", manifest)
        (bundle / "approval.json").write_text("{}")
        (bundle / "resource_policy.rego").write_text("{}")
        bundles[platform] = {
            "build_id": manifest["build_id"],
            "manifest_sha256": hashlib.sha256((bundle / "cvm_manifest.json").read_bytes()).hexdigest(),
        }
    write_json(
        directory / "profile_set.json",
        {"schema_version": 2, "profile_version": "cpu-2026.09", "contract": contract, "bundles": bundles},
    )


@pytest.fixture
def configuration(tmp_path):
    builder = tmp_path / "external builder"
    builder.mkdir()
    (builder / "vault_build.sh").write_text("#!/bin/sh\nexit 1\n")
    (builder / "vault_build.sh").chmod(0o755)
    (builder / "scripts").mkdir()
    (builder / "scripts/cvm_pull").write_text("#!/bin/sh\nexit 1\n")
    (builder / "scripts/cvm_pull").chmod(0o755)
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    write_profile(profile_dir)
    write_docker_archive(tmp_path / "image.tar")
    for name in ("ca.pem", "builder.pem", "builder.key"):
        (tmp_path / name).write_text("input")
    (tmp_path / "cvm_project.yml").write_text(
        yaml.safe_dump(
            {
                "key_service": {
                    "url": "https://keys.example.com:9443",
                    "ca": "ca.pem",
                    "cert": "builder.pem",
                    "key": "builder.key",
                }
            }
        )
    )
    settings = {
        "cvm_builder_dir": "external builder",
        "cvm_image": "profile",
        "docker_archive": "image.tar",
        "output_root": "vault-builds",
        "participants": ["site-1"],
        "platforms": ["intel_tdx"],
        "requires_gpu": False,
    }
    config = {
        "api_version": 3,
        "name": "project1",
        "description": "test project",
        "participants": [
            {"name": "server.example.com", "type": "server", "org": "org", "fed_learn_port": 9002, "admin_port": 9003},
            {
                "name": "site-1",
                "type": "client",
                "org": "org",
                "connect_to": {"host": "server.example.com", "port": 9102},
            },
            {"name": "site-2", "type": "client", "org": "org"},
            {"name": "admin@example.com", "type": "admin", "org": "org", "role": "project_admin"},
        ],
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder"},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
        "cvm_vault": settings,
    }
    return config


def run_provision(config, tmp_path):
    return provision(
        SimpleNamespace(gen_scripts=False),
        copy.deepcopy(config),
        str(tmp_path / "project.yml"),
        str(tmp_path / "workspace"),
    )


def make_adapter(config, tmp_path):
    return VaultAdapter(
        config["cvm_vault"], tmp_path / "project.yml", tmp_path / "workspace", prepare_project(copy.deepcopy(config))
    )


def fake_build(builder_dir, config_file, output, log_file, project_config):
    app = yaml.safe_load(config_file.read_text())
    assert not {"deployment_id", "cvm_profile", "key_service"} & app.keys()
    assert project_config.is_absolute() and project_config.is_file()
    output.mkdir(mode=0o700)
    log_file.write_text("build complete\n")
    deployment = uuid.uuid4().hex
    copies, artifacts = [], {}
    for platform in app.get("platforms", ["amd_sev_snp", "intel_tdx"]):
        # Inventory filenames are deliberately unrelated to deployment/platform labels.
        name = "delivery-" + uuid.uuid4().hex + ".oci.tar"
        identity = {"platform": platform, "deployment_id": deployment, "cvm_build_id": "generic-" + platform}
        config_raw = json.dumps(identity).encode()
        config_hash = hashlib.sha256(config_raw).hexdigest()
        manifest_raw = json.dumps(
            {"artifactType": adapter_module.DELIVERY_TYPE, "config": {"digest": "sha256:" + config_hash}}
        ).encode()
        manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
        write_tar(
            output / name,
            {
                "index.json": json.dumps({"manifests": [{"digest": "sha256:" + manifest_hash}]}).encode(),
                "blobs/sha256/" + manifest_hash: manifest_raw,
                "blobs/sha256/" + config_hash: config_raw,
            },
        )
        artifacts[name] = {
            "artifact_type": adapter_module.DELIVERY_TYPE,
            "manifest_digest": "sha256:" + manifest_hash,
            "archive_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
        }
        copies.append(dict(identity, resource=f"keys/generic-{platform}/binding"))
    write_json(output / "vault_set.json", {"schema_version": 2, "deployment_id": deployment, "copies": copies})
    write_json(output / "oci_artifacts.json", {"schema_version": 1, "artifacts": artifacts})


def test_opt_in_finalized_signed_isolated_workspace(configuration, tmp_path, monkeypatch):
    observed = []
    configuration["participants"][1]["listening_host"] = {"port": 9200}

    def build(builder, config_file, output, log, project_config):
        prod = tmp_path / "workspace/project1/prod_00"
        assert prod.is_dir()
        assert not (tmp_path / "workspace/project1/wip").exists()
        app = yaml.safe_load(config_file.read_text())
        staged = Path(app["application_files"]) / "workspace"
        original = prod / "site-1"
        assert verify_folder_signature(
            str(staged),
            str(staged / "startup/rootCA.pem"),
            single_signer=True,
            signature_file=ProvFileName.SIGNATURE_JSON,
        )
        # StaticFileBuilder writes this file in finalize(), after ordinary signing.
        assert (staged / "local" / ProvFileName.COMM_CONFIG).is_file()
        original_files = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
        staged_files = {p.relative_to(staged): p.read_bytes() for p in staged.rglob("*") if p.is_file()}
        assert original_files == staged_files
        assert sorted(p.name for p in staged.parent.iterdir()) == ["runtime", "workspace"]
        assert (staged / "startup/client.key").is_file()
        assert not (staged / "state").exists()
        assert "user_config" not in app and "user_data" not in app
        for name in ("cvm_image", "docker_archive", "application_files"):
            assert Path(app[name]).is_absolute()
        assert project_config == tmp_path / "cvm_project.yml"
        assert app["image_id"] == docker_image_id(tmp_path / "image.tar")
        assert app["container"]["command"][-2:] == ["--verify", "--foreground"]
        assert app["allowed_ports"] == [9200]
        assert app["allowed_out_ports"] == [443, 8443, 9002, 9003, 9102]
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(config_file.parent.stat().st_mode) == 0o700
        observed.append(app)
        fake_build(builder, config_file, output, log, project_config)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    ctx = run_provision(configuration, tmp_path)
    assert ctx[CtxKey.PROVISION_SUCCESS] is True
    result = ctx[CtxKey.CVM_VAULT_RESULTS][0]
    assert result["participant"] == "site-1"
    assert result["artifacts"][0]["cvm_build_id"] == "generic-intel_tdx"
    assert "key_service" not in json.dumps(result)
    assert not (Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "site-2/signature.json").exists()
    assert len(observed) == 1


@pytest.mark.parametrize("registry", [False, True])
@pytest.mark.parametrize("participant", ["server.example.com", "site-1"])
def test_inputs_match_included_builder(configuration, tmp_path, monkeypatch, registry, participant):
    from nvflare.lighter.cc.image_builder.builder import config as builder_config

    builder = Path(adapter_module.__file__).parent / "image_builder"
    settings = configuration["cvm_vault"]
    settings["cvm_builder_dir"] = str(builder)
    settings["participants"] = [participant]
    del settings["platforms"]
    if registry:
        settings["cvm_image"] = REGISTRY_IMAGE
    observed = []

    def validate_build(builder_dir, config_file, output, log, project_config):
        assert builder_dir == builder.resolve()
        app = builder_config.application(config_file)
        project = builder_config.project(config_file, project_config)
        assert "platforms" not in app
        assert app["image_id"] == docker_image_id(tmp_path / "image.tar")
        assert app["cvm_image"] == (REGISTRY_IMAGE if registry else str(tmp_path / "profile"))
        assert project["key_service"]["key"] == str(tmp_path / "builder.key")
        assert app["container"]["command"][-2:] == ["--verify", "--foreground"]
        observed.append(app)
        fake_build(builder_dir, config_file, output, log, project_config)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", validate_build)
    ctx = run_provision(configuration, tmp_path)
    assert ctx[CtxKey.PROVISION_SUCCESS]
    assert len(observed) == 1
    assert ctx[CtxKey.CVM_VAULT_RESULTS][0]["participant"] == participant


def test_ordinary_provisioning_does_not_use_adapter(configuration, tmp_path, monkeypatch):
    del configuration["cvm_vault"]
    factory = Mock(side_effect=AssertionError("adapter used"))
    monkeypatch.setattr(adapter_module, "VaultAdapter", factory)
    ctx = run_provision(configuration, tmp_path)
    assert ctx[CtxKey.PROVISION_SUCCESS]
    assert CtxKey.CVM_VAULT_RESULTS not in ctx
    factory.assert_not_called()


@pytest.mark.parametrize("phase", ["initialize", "build", "finalize"])
@pytest.mark.parametrize("registry", [False, True])
def test_caught_provision_failure_never_builds_vault(configuration, tmp_path, monkeypatch, phase, registry):
    class FailingBuilder(Builder):
        pass

    def fail(*args):
        raise RuntimeError("provision failure")

    setattr(FailingBuilder, phase, fail)
    if registry:
        configuration["cvm_vault"]["cvm_image"] = REGISTRY_IMAGE
    pull = Mock()
    monkeypatch.setattr(adapter_module.subprocess, "run", pull)
    monkeypatch.setattr(
        "nvflare.lighter.provision.prepare_builders", lambda config: [WorkspaceBuilder(), FailingBuilder()]
    )
    invoke = Mock()
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", invoke)
    existing = tmp_path / "workspace/project1/prod_00"
    existing.mkdir(parents=True)
    (existing / "keep").write_text("pre-existing")
    with pytest.raises(RuntimeError, match="no CVM vaults were built"):
        run_provision(configuration, tmp_path)
    assert (existing / "keep").read_text() == "pre-existing"
    invoke.assert_not_called()
    pull.assert_not_called()


def test_no_new_prod_directory_is_not_success(configuration, tmp_path, monkeypatch):
    (tmp_path / "workspace/project1/prod_99").mkdir(parents=True)
    invoke = Mock()
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", invoke)
    with pytest.raises(RuntimeError, match="no CVM vaults were built"):
        run_provision(configuration, tmp_path)
    invoke.assert_not_called()


@pytest.mark.parametrize(
    "update, message",
    [
        ({"participants": ["missing"]}, "existing clients or servers"),
        ({"participants": ["admin@example.com"]}, "existing clients or servers"),
        ({"participants": []}, "explicitly list"),
        ({"participants": ["site-1", "site-1"]}, "explicitly list"),
        ({"candidate": True}, "Unknown cvm_vault"),
        ({"image_id": "latest"}, "Unknown cvm_vault"),
        ({"builder_dir": "external builder"}, "Unknown cvm_vault"),
        ({"key_service": {}}, "Unknown cvm_vault"),
        ({"requires_gpu": True}, "match the CVM profile"),
        ({"platforms": ["wrong"]}, "Requested platforms"),
        ({"allowed_ports": [True]}, "Ports must be integers"),
        ({"docker_archive": "missing.tar"}, "Missing cvm_vault input"),
        ({"cvm_image": "missing-folder"}, "Missing cvm_vault input"),
        ({"cvm_image": "profile/profile_set.json"}, "pulled directory"),
        ({"cvm_image": "oci://registry.example.org/cvm/cpu:latest"}, "registry references must use"),
        ({"cvm_image": "registry.example.org/cvm/cpu:latest"}, "registry references must use"),
        ({"cvm_image": "https://registry.example.org/cvm/cpu@sha256:bad"}, "registry references must use"),
        ({"cvm_image": "http://" + REGISTRY_IMAGE}, "must use oci:// or https://"),
        ({"cvm_image": "oci://user:secret@" + REGISTRY_IMAGE}, "registry references must use"),
        ({"cvm_profile": "profile/profile_set.json"}, "Use cvm_image"),
        ({"output_root": "workspace/project1/prod_00/vaults"}, "outside the provisioning"),
        ({"release_id": "a.b"}, "Unknown cvm_vault"),
        ({"participant_overrides": {"site-2": {}}}, "selected participants"),
    ],
)
def test_invalid_configuration_fails_before_build(configuration, tmp_path, update, message):
    configuration["cvm_vault"].update(update)
    with pytest.raises(ValueError, match=message):
        make_adapter(configuration, tmp_path)
    assert not (tmp_path / "vault-builds").exists()


def test_multi_participant_overrides(configuration, tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    (public / "data.txt").write_text("public input")
    settings = configuration["cvm_vault"]
    settings["participants"] = ["site-1", "server.example.com"]
    settings["participant_overrides"] = {
        "server.example.com": {
            "platforms": ["amd_sev_snp", "intel_tdx"],
            "user_data": "public",
            "allowed_ports": [1234],
            "allowed_out_ports": [9443],
            "tee_device": True,
        }
    }
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", fake_build)
    ctx = run_provision(configuration, tmp_path)
    assert len(ctx[CtxKey.CVM_VAULT_RESULTS]) == 2
    server = ctx[CtxKey.CVM_VAULT_RESULTS][1]
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", server["deployment_id"])
    assert len(server["artifacts"]) == 2
    app = yaml.safe_load(
        Path(server["output_dir"])
        .with_name(Path(server["output_dir"]).name + "-inputs")
        .joinpath("vault_build.yml")
        .read_text()
    )
    assert app["allowed_ports"] == [1234, 9002, 9003]
    assert app["container"]["ports"] == [{"host": p, "container": p} for p in [1234, 9002, 9003]]
    assert app["container"]["tee_device"] is True
    assert 9443 in app["allowed_out_ports"]
    assert app["user_data"] == str(public)


@pytest.mark.parametrize(
    "name, content",
    [
        ("client.key", b"secret"),
        ("data.pem", b"-----BEGIN PRIVATE KEY-----"),
        ("data.pem", b"a" * (1024 * 1024 - 12) + b"-----BEGIN RSA PRIVATE KEY-----"),
    ],
)
def test_private_keys_rejected_in_public_inputs(configuration, tmp_path, name, content):
    public = tmp_path / "public"
    public.mkdir()
    (public / name).write_bytes(content)
    configuration["cvm_vault"]["user_config"] = "public"
    with pytest.raises(ValueError, match="Private keys"):
        make_adapter(configuration, tmp_path)


def test_public_symlink_rejected(configuration, tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "data").symlink_to(tmp_path / "builder.key")
    configuration["cvm_vault"]["user_data"] = "public"
    with pytest.raises(ValueError, match="symbolic link"):
        make_adapter(configuration, tmp_path)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module.sys, "platform", "linux")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("retain")
    with pytest.raises(FileExistsError, match="already exists"):
        invoke_vault_builder(
            tmp_path, tmp_path / "input.yml", output, tmp_path / "build.log", tmp_path / "cvm_project.yml"
        )
    assert (output / "keep").read_text() == "retain"


def test_nonzero_build_keeps_signed_kit_inputs_logs_and_activation(configuration, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module.sys, "platform", "linux")
    calls = []

    def failed_run(argv, **kwargs):
        calls.append(argv)
        output = Path(argv[-1])
        output.mkdir(mode=0o700)
        (output / "intel_tdx").mkdir()
        write_json(output / "intel_tdx/provisioning.json", {"resource": "keys/build/binding", "state": "uploading"})
        kwargs["stdout"].write("upload acknowledgement lost\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(adapter_module.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="Resolve possible key activation"):
        run_provision(configuration, tmp_path)
    assert len(calls) == 1
    output = Path(calls[0][-1])
    inputs = output.with_name(output.name + "-inputs")
    assert json.loads((output / "intel_tdx/provisioning.json").read_text())["state"] == "uploading"
    assert "acknowledgement" in (inputs / "build.log").read_text()
    assert (inputs / "application/workspace/startup/client.key").is_file()
    assert (tmp_path / "workspace/project1/prod_00/site-1/startup/client.key").is_file()
    assert stat.S_IMODE((inputs / "build.log").stat().st_mode) == 0o600
    assert "--candidate" not in calls[0] and "--dev" not in calls[0]
    assert calls[0][-2] == "--output" and Path(calls[0][-1]).is_absolute()


def test_invoke_uses_array_noninteractive_sudo_and_private_log(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module.sys, "platform", "linux")
    monkeypatch.setattr(adapter_module.os, "geteuid", lambda: 1000)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(adapter_module.subprocess, "run", run)
    invoke_vault_builder(
        tmp_path,
        tmp_path / "vault config.yml",
        tmp_path / "output",
        tmp_path / "build.log",
        tmp_path / "cvm_project.yml",
    )
    argv = run.call_args.args[0]
    assert argv == [
        "sudo",
        "-n",
        str(tmp_path / "vault_build.sh"),
        str(tmp_path / "vault config.yml"),
        "--project-config",
        str(tmp_path / "cvm_project.yml"),
        "--output",
        str(tmp_path / "output"),
    ]
    assert "shell" not in run.call_args.kwargs


@pytest.mark.parametrize("damage", ["platform", "missing", "checksum", "build_id"])
def test_bad_metadata_fails_without_deleting_outputs(configuration, tmp_path, monkeypatch, damage):
    def build(*args):
        fake_build(*args)
        output = args[2]
        if damage in ("platform", "build_id"):
            path = output / "vault_set.json"
            data = json.loads(path.read_text())
            data["copies"][0]["platform" if damage == "platform" else "cvm_build_id"] = "wrong"
            write_json(path, data)
        else:
            archive = next(output.glob("*.oci.tar"))
            archive.unlink() if damage == "missing" else archive.write_bytes(b"corrupt")

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    with pytest.raises(RuntimeError, match="keys may already be active"):
        run_provision(configuration, tmp_path)
    assert list((tmp_path / "vault-builds").glob("*/vault_set.json"))


def test_new_invocation_reuses_same_cvm_ids(configuration, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", fake_build)
    before = {p: p.read_bytes() for p in (tmp_path / "profile").rglob("*") if p.is_file()}
    first = run_provision(copy.deepcopy(configuration), tmp_path)
    second = run_provision(configuration, tmp_path)
    assert first[CtxKey.CVM_VAULT_RESULTS][0]["deployment_id"] != second[CtxKey.CVM_VAULT_RESULTS][0]["deployment_id"]
    assert (
        first[CtxKey.CVM_VAULT_RESULTS][0]["artifacts"][0]["cvm_build_id"]
        == second[CtxKey.CVM_VAULT_RESULTS][0]["artifacts"][0]["cvm_build_id"]
    )
    assert all(p.read_bytes() == content for p, content in before.items())


def test_loopback_server_address_fails_before_build(configuration, tmp_path, monkeypatch):
    configuration["participants"][1]["connect_to"] = {"host": "localhost", "port": 9002}
    invoke = Mock()
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", invoke)
    with pytest.raises(ValueError, match="reachable server"):
        run_provision(configuration, tmp_path)
    invoke.assert_not_called()


def test_gpu_profile_override_does_not_change_cpu_participant(configuration, tmp_path, monkeypatch):
    shutil.copytree(tmp_path / "profile", tmp_path / "gpu-profile")
    write_profile(tmp_path / "gpu-profile", gpu="nvidia_cc")
    settings = configuration["cvm_vault"]
    settings["participants"] = ["site-1", "site-2"]
    settings["participant_overrides"] = {
        "site-2": {
            "cvm_image": "gpu-profile",
            "requires_gpu": True,
            "platforms": ["amd_sev_snp"],
        }
    }
    apps = []

    def build(*args):
        apps.append(yaml.safe_load(args[1].read_text()))
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    run_provision(configuration, tmp_path)
    assert apps[0]["requires_gpu"] is False
    assert apps[1]["requires_gpu"] is True
    assert apps[1]["platforms"] == ["amd_sev_snp"]


def test_stale_context_cannot_reuse_existing_production_directory(configuration, tmp_path):
    prod = tmp_path / "workspace/project1/prod_00"
    prod.mkdir(parents=True)
    adapter = make_adapter(configuration, tmp_path)
    with pytest.raises(ValueError, match="new production directory"):
        adapter.build({CtxKey.PROVISION_SUCCESS: True, CtxKey.CURRENT_PROD_DIR: str(prod)})


def test_one_participant_failure_preserves_earlier_success(configuration, tmp_path, monkeypatch):
    configuration["cvm_vault"]["participants"] = ["site-1", "site-2"]
    calls = []

    def build(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("second participant failed")
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    with pytest.raises(RuntimeError, match="second participant"):
        run_provision(configuration, tmp_path)
    assert len(calls) == 2
    assert (calls[0][1].parent / "result.json").is_file()
    assert list(calls[0][2].glob("*.oci.tar"))
    assert calls[1][1].is_file()


@pytest.mark.parametrize("prefix", ["oci://", "https://", ""])
def test_registry_reference_passed_intact_to_builder(configuration, tmp_path, monkeypatch, prefix):
    settings = configuration["cvm_vault"]
    settings["cvm_image"] = prefix + REGISTRY_IMAGE
    del settings["platforms"]
    calls = []

    def build(*args):
        app = yaml.safe_load(args[1].read_text())
        assert app["cvm_image"] == prefix + REGISTRY_IMAGE
        assert "platforms" not in app
        calls.append(args)
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    ctx = run_provision(configuration, tmp_path)
    assert len(calls) == 1
    assert len(ctx[CtxKey.CVM_VAULT_RESULTS][0]["artifacts"]) == 2


@pytest.mark.parametrize("absolute", [False, True])
def test_local_image_resolves_folder_without_pull(configuration, tmp_path, monkeypatch, absolute):
    configuration["cvm_vault"]["cvm_image"] = str(tmp_path / "profile") if absolute else "./profile"
    pull = Mock(side_effect=AssertionError("Local images must not pull"))
    monkeypatch.setattr(adapter_module.subprocess, "run", pull)
    monkeypatch.setattr(adapter_module, "invoke_vault_builder", fake_build)
    ctx = run_provision(configuration, tmp_path)
    assert ctx[CtxKey.CVM_VAULT_RESULTS]
    pull.assert_not_called()


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("mode", ["w", "w:gz"])
def test_image_id_derived_from_config_bytes(tmp_path, modern, mode):
    archive = tmp_path / "application.tar"
    expected = write_docker_archive(archive, modern=modern, mode=mode)
    assert docker_image_id(archive) == expected


def test_duplicate_tags_for_same_image_are_unambiguous(tmp_path):
    archive = tmp_path / "application.tar"
    expected = write_docker_archive(archive)
    with tarfile.open(archive) as tar:
        config = tar.extractfile(expected[7:] + ".json").read()
    write_docker_archive(archive, extra=config)
    assert docker_image_id(archive) == expected


@pytest.mark.parametrize(
    "damage", ["multiple", "wrong_arch", "wrong_os", "empty", "export", "link", "duplicate", "traversal"]
)
def test_ambiguous_or_invalid_docker_archive_rejected(tmp_path, damage):
    archive = tmp_path / "image.tar"
    if damage == "multiple":
        write_docker_archive(archive, extra=json.dumps({"os": "linux", "architecture": "amd64"}).encode())
    elif damage in ("wrong_arch", "wrong_os"):
        write_docker_archive(
            archive,
            architecture="arm64" if damage == "wrong_arch" else "amd64",
            os_name="windows" if damage == "wrong_os" else "linux",
        )
    elif damage in ("empty", "export", "traversal"):
        write_tar(
            archive,
            (
                {}
                if damage == "export"
                else {"manifest.json": json.dumps([] if damage == "empty" else [{"Config": "../config.json"}]).encode()}
            ),
        )
    else:
        write_docker_archive(archive)
        with tarfile.open(archive, "a") as tar:
            member = tarfile.TarInfo("manifest.json")
            if damage == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            tar.addfile(member)
    with pytest.raises(ValueError, match="Invalid docker_archive"):
        docker_image_id(archive)


def test_omitted_platforms_stays_omitted(configuration, tmp_path, monkeypatch):
    del configuration["cvm_vault"]["platforms"]
    apps = []

    def build(*args):
        apps.append(yaml.safe_load(args[1].read_text()))
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    ctx = run_provision(configuration, tmp_path)
    assert "platforms" not in apps[0]
    assert len(ctx[CtxKey.CVM_VAULT_RESULTS][0]["artifacts"]) == 2


@pytest.mark.parametrize("value", [None, [], "intel_tdx", ["intel_tdx", "intel_tdx"]])
def test_explicit_invalid_platforms_rejected(configuration, tmp_path, value):
    configuration["cvm_vault"]["platforms"] = value
    with pytest.raises(ValueError, match="Requested platforms"):
        make_adapter(configuration, tmp_path)


def test_default_output_matches_previous_participant_folder(configuration, tmp_path, monkeypatch):
    del configuration["cvm_vault"]["output_root"]
    configuration["cvm_vault"]["participants"] = ["site-1", "server.example.com"]
    calls = []

    def build(*args):
        assert not args[2].exists()
        original = args[1].parent / "startup-kit"
        assert verify_folder_signature(
            str(original),
            str(original / "startup/rootCA.pem"),
            single_signer=True,
            signature_file=ProvFileName.SIGNATURE_JSON,
        )
        calls.append(args)
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    ctx = run_provision(configuration, tmp_path)
    for result in ctx[CtxKey.CVM_VAULT_RESULTS]:
        expected = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / result["participant"]
        assert result["output_dir"] == str(expected)
        assert all(Path(a["path"]).parent == expected for a in result["artifacts"])
    assert all(call[1].parent.parent == tmp_path / "workspace/project1/.cvm-vault-builds" for call in calls)
    assert (Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "site-2/startup/client.key").is_file()


def test_default_output_failure_retains_original_signed_kit(configuration, tmp_path, monkeypatch):
    del configuration["cvm_vault"]["output_root"]
    calls = []

    def failed(*args):
        calls.append(args)
        args[2].mkdir()
        (args[2] / "build_failure.json").write_text('{"upload_acknowledged":true}')
        raise RuntimeError("packaging failed")

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", failed)
    with pytest.raises(RuntimeError, match="packaging failed"):
        run_provision(configuration, tmp_path)
    assert len(calls) == 1
    assert (calls[0][1].parent / "startup-kit/startup/client.key").is_file()
    assert (calls[0][2] / "build_failure.json").is_file()


def test_project_config_discovery_uses_project_location(configuration, tmp_path, monkeypatch):
    nested = tmp_path / "nested/project"
    nested.mkdir(parents=True)
    (nested / "project.yml").write_text("# project")
    settings = copy.deepcopy(configuration["cvm_vault"])
    for key in ("cvm_builder_dir", "cvm_image", "docker_archive", "output_root"):
        settings[key] = str(tmp_path / settings[key])
    monkeypatch.chdir(tmp_path / "external builder")
    adapter = VaultAdapter(settings, nested / "project.yml", tmp_path / "workspace", prepare_project(configuration))
    assert adapter.project_config == tmp_path / "cvm_project.yml"


def test_project_config_shared_and_never_staged(configuration, tmp_path, monkeypatch):
    configuration["cvm_vault"]["participants"] = ["site-1", "site-2"]
    configs = []

    def build(*args):
        configs.append(args[4])
        assert "key_service" not in yaml.safe_load(args[1].read_text())
        assert not list(args[1].parent.rglob("builder.key"))
        fake_build(*args)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    run_provision(configuration, tmp_path)
    assert configs == [tmp_path / "cvm_project.yml"] * 2


@pytest.mark.parametrize("damage", ["missing", "invalid", "credential", "url", "null_explicit"])
def test_bad_project_config_fails_before_provisioning(configuration, tmp_path, damage):
    path = tmp_path / "cvm_project.yml"
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("unexpected: true\n")
    elif damage == "credential":
        (tmp_path / "builder.key").unlink()
    elif damage == "url":
        path.write_text(path.read_text().replace("https:", "http:"))
    else:
        configuration["cvm_vault"]["project_config"] = None
    with pytest.raises(ValueError):
        make_adapter(configuration, tmp_path)
    assert not (tmp_path / "workspace").exists()


def test_explicit_project_config_overrides_invalid_nearest(configuration, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    config = yaml.safe_load((tmp_path / "cvm_project.yml").read_text())
    for key in ("ca", "cert", "key"):
        config["key_service"][key] = "../" + config["key_service"][key]
    (other / "selected.yml").write_text(yaml.safe_dump(config))
    (tmp_path / "cvm_project.yml").write_text("invalid: true\n")
    configuration["cvm_vault"]["project_config"] = "other/selected.yml"
    assert make_adapter(configuration, tmp_path).project_config == other / "selected.yml"
    configuration["cvm_vault"]["project_config"] = "missing.yml"
    with pytest.raises(ValueError, match="Missing cvm_vault input"):
        make_adapter(configuration, tmp_path)


def test_invalid_nearest_project_does_not_fall_back(configuration, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "cvm_project.yml").write_text("invalid: true\n")
    settings = copy.deepcopy(configuration["cvm_vault"])
    for key in ("cvm_builder_dir", "cvm_image", "docker_archive", "output_root"):
        settings[key] = str(tmp_path / settings[key])
    with pytest.raises(ValueError, match="only key_service"):
        VaultAdapter(settings, nested / "project.yml", tmp_path / "workspace", prepare_project(configuration))


@pytest.mark.parametrize("damage", ["deployment", "copy_deployment", "filename", "duplicate_platform", "oci_digest"])
def test_output_identity_and_inventory_are_checked(configuration, tmp_path, monkeypatch, damage):
    def build(*args):
        fake_build(*args)
        output = args[2]
        path = output / ("oci_artifacts.json" if damage in ("filename", "oci_digest") else "vault_set.json")
        data = json.loads(path.read_text())
        if damage == "deployment":
            data["deployment_id"] = uuid.uuid4().hex
        elif damage == "copy_deployment":
            data["copies"][0]["deployment_id"] = uuid.uuid4().hex
        elif damage == "filename":
            name = next(iter(data["artifacts"]))
            data["artifacts"]["../" + name] = data["artifacts"].pop(name)
        elif damage == "duplicate_platform":
            data["copies"].append(data["copies"][0])
        else:
            next(iter(data["artifacts"].values()))["manifest_digest"] = "sha256:" + "0" * 64
        write_json(path, data)

    monkeypatch.setattr(adapter_module, "invoke_vault_builder", build)
    with pytest.raises(RuntimeError, match="keys may already be active"):
        run_provision(configuration, tmp_path)
    assert list((tmp_path / "vault-builds").glob("*/vault_set.json"))
