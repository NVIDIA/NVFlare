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

"""Existing-worker identity and collector publication regressions."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

tomllib = pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "examples/devops/coco/trusted_system/lib/installed-platform.py"
spec = importlib.util.spec_from_file_location("installed_platform", SOURCE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
IMAGE = "registry.example.com/coco/collector@sha256:" + "a" * 64


def collector(tmp_path):
    metadata = {"schema": 1, "image": IMAGE, "image_id": "sha256:" + "b" * 64, "cpu_tee": "amd_sev_snp"}
    path = tmp_path / "collector-image.json"
    path.write_text(json.dumps(metadata))
    ca = tmp_path / "ca.crt"
    ca.write_text("public registry CA")
    env = {
        "COLLECTOR_METADATA_FILE": str(path),
        "COLLECTOR_METADATA_SHA256": workflow.digest(path),
        "COLLECTOR_IMAGE": IMAGE,
        "RUNTIME_CLASS": "kata-qemu-snp",
        "COLLECTOR_REGISTRY_CA": str(ca),
    }
    return metadata, env


def test_supplied_collector_checksum_is_checked_before_acceptance(tmp_path):
    metadata, env = collector(tmp_path)
    assert workflow.load_collector(env) == metadata
    Path(env["COLLECTOR_METADATA_FILE"]).write_text(json.dumps(dict(metadata, image=IMAGE.replace("a", "c"))))
    with pytest.raises(ValueError, match="supplied SHA256"):
        workflow.load_collector(env)


@pytest.mark.parametrize("supplied", [None, ""])
def test_collector_handoff_does_not_require_a_checksum(tmp_path, supplied):
    metadata, env = collector(tmp_path)
    if supplied is None:
        env.pop("COLLECTOR_METADATA_SHA256")
    else:
        env["COLLECTOR_METADATA_SHA256"] = supplied
    assert workflow.load_collector(env) == metadata
    checked, actual = workflow.load_collector(env, with_checksum=True)
    assert checked == metadata
    assert actual == workflow.digest(env["COLLECTOR_METADATA_FILE"])


def test_existing_collector_metadata_accepts_legacy_cpu_tee_name(tmp_path):
    metadata, env = collector(tmp_path)
    metadata["cpu_tee"] = "snp"
    path = Path(env["COLLECTOR_METADATA_FILE"])
    path.write_text(json.dumps(metadata))
    env["COLLECTOR_METADATA_SHA256"] = workflow.digest(path)

    assert workflow.load_collector(env) == metadata


@pytest.mark.parametrize("supplied", ["A" * 64, "b" * 63, 123, False, 0])
def test_malformed_supplied_collector_checksum_is_rejected(tmp_path, supplied):
    _, env = collector(tmp_path)
    env["COLLECTOR_METADATA_SHA256"] = supplied
    with pytest.raises(ValueError, match="when supplied"):
        workflow.load_collector(env)


@pytest.mark.parametrize(
    "key,value",
    [
        ("COLLECTOR_IMAGE", "registry.example.com/collector:latest"),
        ("COLLECTOR_METADATA_SHA256", "not-a-sha256"),
        ("RUNTIME_CLASS", "kata-qemu-tdx"),
    ],
)
def test_collector_target_or_reference_change_is_rejected(tmp_path, key, value):
    _, env = collector(tmp_path)
    env[key] = value
    with pytest.raises(ValueError):
        workflow.load_collector(env)


def worker_snapshot():
    return {
        "node": {
            "name": "selected-worker",
            "uid": "node-uid",
            "machine_id": "machine-id",
            "addresses": [{"type": "InternalIP", "address": "192.0.2.1"}],
        },
        "kubeconfig": "/operator/kubeconfig",
    }


def worker_object(snapshot):
    return {
        "metadata": {"name": snapshot["node"]["name"], "uid": snapshot["node"]["uid"]},
        "status": {
            "nodeInfo": {"machineID": snapshot["node"]["machine_id"]},
            "addresses": snapshot["node"]["addresses"],
        },
    }


def test_local_worker_is_identified_by_machine_id_and_address_not_hostname():
    snapshot = worker_snapshot()
    addresses = [{"addr_info": [{"local": "192.0.2.1"}]}]
    with (
        patch.object(
            workflow.subprocess,
            "check_output",
            side_effect=[json.dumps(worker_object(snapshot)), json.dumps(addresses)],
        ),
        patch.object(Path, "read_text", return_value="machine-id\n"),
    ):
        assert workflow.verify_worker(snapshot, ["kubectl"]) == snapshot["node"]


@pytest.mark.parametrize("change", ["uid", "machine", "address"])
def test_changed_or_nonlocal_worker_is_rejected(change):
    snapshot = worker_snapshot()
    node = worker_object(snapshot)
    if change == "uid":
        node["metadata"]["uid"] = "replacement-node"
    machine = "other-machine" if change == "machine" else "machine-id"
    local = "192.0.2.99" if change == "address" else "192.0.2.1"
    with (
        patch.object(
            workflow.subprocess,
            "check_output",
            side_effect=[json.dumps(node), json.dumps([{"addr_info": [{"local": local}]}])],
        ),
        patch.object(Path, "read_text", return_value=machine),
    ):
        with pytest.raises(ValueError):
            workflow.verify_worker(snapshot, ["kubectl"])


def test_collector_and_launch_must_stay_on_the_inspected_worker(tmp_path):
    snapshot = worker_snapshot()
    path = tmp_path / "installed-platform.json"
    path.write_text(json.dumps(snapshot))
    pod = {"spec": {"nodeName": snapshot["node"]["name"]}}
    workflow.check_pod_node(pod, snapshot)
    launch = {
        "node_name": snapshot["node"]["name"],
        "node_uid": snapshot["node"]["uid"],
        "platform_snapshot_sha256": workflow.digest(path),
    }
    workflow.verify_launch(launch, snapshot, tmp_path)
    with pytest.raises(ValueError):
        workflow.check_pod_node({"spec": {"nodeName": "other-worker"}}, snapshot)
    launch["node_uid"] = "changed-node"
    with pytest.raises(ValueError):
        workflow.verify_launch(launch, snapshot, tmp_path)


def runtime_snapshot(tmp_path):
    # The ready cluster may use any local installation directory; no upstream
    # image, chart, artifact copy or installer provenance is required.
    installed_root = tmp_path / "existing-runtime"
    installed_root.mkdir()
    artifacts = {}
    for key in ("path", "firmware", "kernel", "initrd"):
        artifact = installed_root / key
        artifact.write_text("existing installed " + key)
        artifacts[key] = str(artifact)
    config = tmp_path / "installed.toml"
    config.write_text(
        '[hypervisor.qemu]\nkernel_params = "quiet agent.guest_components_rest_api=all"\n'
        "confidential_guest = true\nsev_snp_guest = true\ndefault_vcpus = 1\ndefault_memory = 8192\n"
        + "".join(key + " = " + json.dumps(value) + "\n" for key, value in artifacts.items())
    )
    parsed = tomllib.loads(config.read_text())
    snapshot = worker_snapshot()
    snapshot.update(
        installed_config=str(config),
        runtime_class="kata-qemu-snp",
        config=parsed,
        files={str(config): workflow.digest(config), **{value: workflow.digest(value) for value in artifacts.values()}},
    )
    (tmp_path / "installed-platform.json").write_text(json.dumps(snapshot))
    return snapshot, config


def test_snapshot_rechecks_current_runtime_bytes_and_derivation(tmp_path):
    snapshot, installed = runtime_snapshot(tmp_path)
    with patch.object(workflow, "verify_worker"):
        assert workflow.verify_snapshot(tmp_path) == snapshot
        installed.write_text(installed.read_text() + "# changed after capture\n")
        with pytest.raises(ValueError, match="changed after inspection"):
            workflow.verify_snapshot(tmp_path)


def test_missing_guest_token_api_cannot_be_approved_even_with_updated_host_hash(tmp_path):
    snapshot, installed = runtime_snapshot(tmp_path)
    installed.write_text(installed.read_text().replace(" agent.guest_components_rest_api=all", ""))
    snapshot["files"][str(installed)] = workflow.digest(installed)
    (tmp_path / "installed-platform.json").write_text(json.dumps(snapshot))
    with patch.object(workflow, "verify_worker"), pytest.raises(ValueError, match="guest_components_rest_api=all"):
        workflow.verify_snapshot(tmp_path)


def test_existing_registry_cache_uses_only_public_inputs_and_the_digest(tmp_path):
    metadata, env = collector(tmp_path)
    outputs = [
        json.dumps({"Digest": IMAGE.rsplit("@", 1)[1]}),
        json.dumps({"config": {"digest": metadata["image_id"]}}),
    ]
    with (
        patch.object(workflow.subprocess, "check_output", side_effect=outputs) as inspect,
        patch.object(workflow.subprocess, "run") as run,
    ):
        workflow.cache_collector(IMAGE, tmp_path, env)
    commands = [call.args[0] for call in run.call_args_list]
    assert not any(command[0] == "docker" for command in commands)
    assert "docker://" + IMAGE in commands[0]
    assert "--src-tls-verify=true" in commands[0]
    assert "--dest-tls-verify=true" not in commands[0]  # destination is a local archive
    assert all("--tls-verify=true" in call.args[0] for call in inspect.call_args_list)
    assert commands[-1][-1] == IMAGE
    assert not any("password" in value or "dest-authfile" in value for command in commands for value in command)


def test_mismatched_registry_config_stops_before_cache_import(tmp_path):
    _, env = collector(tmp_path)
    outputs = [
        json.dumps({"Digest": IMAGE.rsplit("@", 1)[1]}),
        json.dumps({"config": {"digest": "sha256:" + "c" * 64}}),
    ]
    with (
        patch.object(workflow.subprocess, "check_output", side_effect=outputs),
        patch.object(workflow.subprocess, "run") as run,
    ):
        with pytest.raises(ValueError, match="configuration differs"):
            workflow.cache_collector(IMAGE, tmp_path, env)
        run.assert_not_called()


def test_publication_uses_owner_auth_but_checks_anonymous_readback(tmp_path):
    _, env = collector(tmp_path)
    auth = tmp_path / "publisher-auth.json"
    auth.write_text('{"auths":{}}')
    auth.chmod(0o600)
    env["COLLECTOR_REGISTRY_AUTH_FILE"] = str(auth)
    digest = "sha256:" + "a" * 64

    def publish(command, **kwargs):
        Path(command[command.index("--digestfile") + 1]).write_text(digest)
        return subprocess.CompletedProcess(command, 0)

    with (
        patch.object(workflow.subprocess, "run", side_effect=publish) as run,
        patch.object(workflow.subprocess, "check_output", return_value=digest) as inspect,
    ):
        image = workflow.publish_collector(
            "local:collector", "registry.example.com/coco/collector:reviewed", tmp_path, env
        )
    assert image == IMAGE
    assert "--dest-authfile" in run.call_args.args[0]
    assert str(auth) in run.call_args.args[0]
    assert "--dest-tls-verify=true" in run.call_args.args[0]
    assert "--src-tls-verify=true" not in run.call_args.args[0]  # source is the local Docker daemon
    assert "--tls-verify=true" in inspect.call_args.args[0]


@pytest.mark.parametrize("supplied", [None, ""])
@pytest.mark.parametrize("replace_after_read", [False, True])
@pytest.mark.parametrize("observed_version", ["3.29.0", "3.30.0", "3.29.0-rc1"])
def test_snapshot_records_received_metadata_without_a_supplied_checksum(
    tmp_path, supplied, replace_after_read, observed_version
):
    old_snapshot, installed = runtime_snapshot(tmp_path)
    (tmp_path / "installed-platform.json").unlink()
    original, env = collector(tmp_path)
    metadata_source = Path(env["COLLECTOR_METADATA_FILE"])
    received = metadata_source.read_bytes()
    actual_open = Path.open

    class ReplacingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, limit):
            assert limit == 65537
            raw = self.stream.read(limit)
            replacement = dict(original, image_id="sha256:" + "c" * 64)
            with actual_open(metadata_source, "wb") as target:
                target.write(json.dumps(replacement).encode())
            return raw

    def open_metadata(path, *args, **kwargs):
        stream = actual_open(path, *args, **kwargs)
        if replace_after_read and path == metadata_source and args == ("rb",):
            return ReplacingReader(stream)
        return stream

    if supplied is None:
        env.pop("COLLECTOR_METADATA_SHA256")
    else:
        env["COLLECTOR_METADATA_SHA256"] = supplied
    runtime = tmp_path / "kata-runtime"
    runtime.write_text("existing runtime fixture")
    verifier = tmp_path / "snpguest"
    verifier.write_text("reviewed verifier fixture")
    verifier.chmod(0o700)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.touch()
    env.update(
        NODE_NAME=old_snapshot["node"]["name"],
        RUNTIME_CLASS="kata-qemu-snp",
        INSTALLED_KATA_CONFIG=str(installed),
        KATA_RUNTIME=str(runtime),
        KUBECONFIG_PATH=str(kubeconfig),
        SNP_VERIFIER=str(verifier),
    )

    def installed_output(command, **kwargs):
        if command[0] == "kubectl":
            return json.dumps(worker_object(old_snapshot))
        if "kata-env" in command:
            return json.dumps({"Kernel": {"Parameters": "quiet agent.guest_components_rest_api=all"}})
        assert command == [str(runtime), "--version"]
        return "kata-runtime : " + observed_version + "\n"

    with (
        patch.object(workflow, "verify_worker"),
        patch.object(workflow.subprocess, "check_output", side_effect=installed_output),
        patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
        patch.object(Path, "open", open_metadata),
    ):
        if observed_version != "3.29.0":
            with pytest.raises(ValueError, match="compatible with this example"):
                workflow.inspect_platform(tmp_path, env)
        else:
            workflow.inspect_platform(tmp_path, env)
    if observed_version != "3.29.0":
        assert not (tmp_path / "installed-platform.json").exists()
        return
    recorded = json.loads((tmp_path / "installed-platform.json").read_text())
    checksum = hashlib.sha256(received).hexdigest()
    assert recorded["collector_metadata_sha256"] == checksum
    assert recorded["files"][env["COLLECTOR_METADATA_FILE"]] == checksum
    copy_path = Path(recorded["collector_metadata_snapshot"])
    assert copy_path.read_bytes() == received
    assert recorded["files"][str(copy_path)] == checksum
    assert workflow.load_collector(env, profile=tmp_path) == original
    env.update(
        NODE_NAME=old_snapshot["node"]["name"],
        INSTALLED_KATA_CONFIG=str(installed),
        KATA_RUNTIME=str(runtime),
        KUBECONFIG_PATH=str(kubeconfig),
    )
    with patch.object(workflow, "verify_worker"):
        if replace_after_read:
            with pytest.raises(ValueError, match="changed after inspection"):
                workflow.verify_snapshot(tmp_path, env)
            return
        assert workflow.verify_snapshot(tmp_path, env) == recorded
        # Later consumers use the profile copy and check its recorded digest.
        copy_path.write_text(copy_path.read_text() + "\n")
        with pytest.raises(ValueError, match="supplied SHA256"):
            workflow.load_collector(env, profile=tmp_path)
        copy_path.write_bytes(received)
        metadata = Path(env["COLLECTOR_METADATA_FILE"])
        metadata.write_text(metadata.read_text() + "\n")
        with pytest.raises(ValueError, match="changed after inspection"):
            workflow.verify_snapshot(tmp_path, env)
