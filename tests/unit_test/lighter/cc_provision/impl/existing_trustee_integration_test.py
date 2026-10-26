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

"""API-only registration, unrelated-policy preservation and uncertain-commit recovery."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.lighter.cc_provision import workload_release, workload_security

COCO = Path(__file__).resolve().parents[5] / "examples/devops/coco"
SPEC = importlib.util.spec_from_file_location(
    "coco_ready_trustee_receiver", COCO / "service/lib/install-workload-release.py"
)
RECEIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIVER)
IMAGE = "registry.example.com:5443/workload@sha256:" + "a" * 64
INITIAL_POLICY = b'package policy\nimport rego.v1\ndefault allow := false\nallow if { input.fixture == "unrelated" }\n'


def update_manifest(directory):
    manifest = "".join(
        hashlib.sha256((directory / name).read_bytes()).hexdigest() + "  " + name + "\n"
        for name in sorted(workload_security.HANDOFF_FILES)
    )
    (directory / "SHA256SUMS").write_text(manifest)
    return hashlib.sha256(manifest.encode()).hexdigest()


def make_handoff(directory, runtime="kata-qemu-snp", release_name="demo"):
    directory.mkdir(parents=True)
    resources = [f"default/{kind}/{release_name}" for kind in ("image-key", "sig-public-key", "security-policy")]
    auth = workload_release.build_authorization(runtime, release_name, IMAGE, ["/bin/workload"], "b" * 64, resources)
    template = (COCO / "service/policies/workload-resource-policy.rego.template").read_text()
    (directory / "release-authorization.json").write_text(json.dumps(auth))
    (directory / "resource-policy-fragment.rego").write_text(workload_release.render_fragment(auth, template))
    (directory / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nfixture\n-----END PUBLIC KEY-----\n")
    (directory / "image_key").write_bytes(b"k" * 32)
    (directory / "image_key").chmod(0o600)
    (directory / "image-security-policy.json").write_text(
        json.dumps(
            {
                "default": [{"type": "reject"}],
                "transports": {
                    "docker": {
                        IMAGE.split("@")[0]: [
                            {
                                "type": "sigstoreSigned",
                                "keyPath": f"kbs:///default/sig-public-key/{release_name}",
                                "signedIdentity": {"type": "matchRepository"},
                            }
                        ]
                    }
                },
            }
        )
    )
    return directory, update_manifest(directory)


class FakeClient:
    def __init__(self):
        self.events = []
        self.resources = {"default/image-key/unrelated": b"keep-existing-key"}
        self.active_policy = INITIAL_POLICY
        self.fail_resource = None
        self.fail_policy = None

    def set_resource(self, resource, filename):
        self.events.append(("resource", resource))
        if resource.split("/")[1] == self.fail_resource:
            raise ValueError("resource upload failed")
        self.resources[resource] = Path(filename).read_bytes()

    def set_policy(self, filename):
        candidate = Path(filename)
        marker = candidate.with_name(candidate.name.removesuffix(".pending.rego") + ".pending.json")
        assert marker.is_file(), "pending marker must be durable before policy POST"
        self.events.append(("policy", candidate.read_bytes()))
        if self.fail_policy == "before":
            raise ValueError("policy request interrupted")
        self.active_policy = candidate.read_bytes()
        if self.fail_policy == "after":
            raise ValueError("response lost after remote commit")


@pytest.fixture
def registration(tmp_path):
    source, pin = make_handoff(tmp_path / "handoff")
    current = tmp_path / "current-resource-policy.rego"
    current.write_bytes(INITIAL_POLICY)
    return source, pin, current, tmp_path / "state", FakeClient()


@pytest.mark.parametrize("runtime", workload_release.RUNTIME_TARGETS)
@pytest.mark.parametrize("pin_mode", ["omitted", "empty", "valid"])
def test_ready_trustee_registration_preserves_unrelated_rules_and_publishes_policy_last(tmp_path, runtime, pin_mode):
    source, pin = make_handoff(tmp_path / "handoff", runtime)
    current = tmp_path / "current.rego"
    current.write_bytes(INITIAL_POLICY)
    client = FakeClient()
    expected = {"omitted": None, "empty": "", "valid": pin}[pin_mode]
    receipt = RECEIVER.install_release(source, current, tmp_path / "state", client, expected)
    assert [kind for kind, _ in client.events] == ["resource", "resource", "resource", "policy"]
    assert current.read_bytes() == client.active_policy
    assert current.read_bytes().startswith(INITIAL_POLICY.rstrip())
    assert client.resources["default/image-key/unrelated"] == b"keep-existing-key"
    assert receipt["status"] == "uploads-and-policy-accepted"
    assert "persistent" not in receipt and "attestation" not in receipt
    assert not Path(str(current) + ".pending.json").exists()
    assert not Path(str(current) + ".pending.rego").exists()


@pytest.mark.parametrize("kind", ["sig-public-key", "security-policy", "image-key"])
def test_partial_upload_never_activates_policy_and_same_handoff_can_retry(registration, kind):
    source, _, current, state, client = registration
    client.fail_resource = kind
    with pytest.raises(ValueError, match="resource upload failed"):
        RECEIVER.install_release(source, current, state, client)
    assert current.read_bytes() == client.active_policy == INITIAL_POLICY
    assert not any(event[0] == "policy" for event in client.events)
    assert not Path(str(current) + ".pending.json").exists()
    client.fail_resource = None
    RECEIVER.install_release(source, current, state, client)
    assert current.read_bytes() == client.active_policy
    assert client.resources["default/image-key/unrelated"] == b"keep-existing-key"


@pytest.mark.parametrize("failure", ["before", "after"])
def test_uncertain_policy_commit_blocks_other_handoffs_and_retries_identical_candidate(registration, tmp_path, failure):
    source, _, current, state, client = registration
    client.fail_policy = failure
    with pytest.raises(ValueError, match="Policy registration is pending"):
        RECEIVER.install_release(source, current, state, client)
    assert current.read_bytes() == INITIAL_POLICY
    candidate = Path(str(current) + ".pending.rego").read_bytes()
    other, _ = make_handoff(tmp_path / "other-handoff", release_name="another-release")
    count = len(client.events)
    with pytest.raises(ValueError, match="Different releases are blocked"):
        RECEIVER.install_release(other, current, state, client)
    assert len(client.events) == count
    client.fail_policy = None
    RECEIVER.install_release(source, current, state, client)
    assert current.read_bytes() == client.active_policy == candidate
    assert [data for kind, data in client.events if kind == "policy"] == [candidate, candidate]
    assert not Path(str(current) + ".pending.json").exists()


@pytest.mark.parametrize("failed_write", ["canonical", "receipt"])
def test_local_commit_failure_retains_recoverable_pending_candidate(registration, monkeypatch, failed_write):
    source, _, current, state, client = registration
    atomic_write = RECEIVER.atomic_write

    def fail_local_write(path, content):
        if (failed_write == "canonical" and Path(path) == current) or (
            failed_write == "receipt" and Path(path).parent.name == "receipts"
        ):
            raise OSError("fixture local write failure")
        atomic_write(path, content)

    with monkeypatch.context() as scope:
        scope.setattr(RECEIVER, "atomic_write", fail_local_write)
        with pytest.raises(ValueError, match="Policy registration is pending"):
            RECEIVER.install_release(source, current, state, client)
    candidate = Path(str(current) + ".pending.rego").read_bytes()
    assert client.active_policy == candidate
    assert current.read_bytes() == (INITIAL_POLICY if failed_write == "canonical" else candidate)
    RECEIVER.install_release(source, current, state, client)
    assert current.read_bytes() == candidate
    assert not Path(str(current) + ".pending.json").exists()


@pytest.mark.parametrize("change", ["candidate", "canonical", "marker"])
def test_pending_drift_stops_before_further_api_writes(registration, change):
    source, _, current, state, client = registration
    client.fail_policy = "before"
    with pytest.raises(ValueError):
        RECEIVER.install_release(source, current, state, client)
    paths = {
        "candidate": Path(str(current) + ".pending.rego"),
        "canonical": current,
        "marker": Path(str(current) + ".pending.json"),
    }
    paths[change].write_text("unexpected local change")
    count = len(client.events)
    with pytest.raises(ValueError, match="pending"):
        RECEIVER.install_release(source, current, state, client)
    assert len(client.events) == count


def test_policy_aliases_use_one_canonical_lock_and_pending_state(registration, tmp_path):
    source, _, current, state, client = registration
    alias = tmp_path / "alias.rego"
    alias.symlink_to(current)
    client.fail_policy = "after"
    with pytest.raises(ValueError):
        RECEIVER.install_release(source, alias, state, client)
    assert Path(str(current) + ".lock").exists()
    assert Path(str(current) + ".pending.json").exists()
    assert not Path(str(alias) + ".lock").exists()
    client.fail_policy = None
    RECEIVER.install_release(source, current, state, client)
    assert alias.read_bytes() == current.read_bytes() == client.active_policy


@pytest.mark.parametrize("pin", ["bad", "0" * 64])
def test_supplied_invalid_or_mismatched_pin_fails_before_upload(registration, pin):
    source, _, current, state, client = registration
    with pytest.raises(ValueError, match="manifest pin"):
        RECEIVER.install_release(source, current, state, client, pin)
    assert not client.events


@pytest.mark.parametrize("change", ["extra-file", "payload", "fragment", "image-policy"])
def test_handoff_integrity_and_template_checks_precede_api_writes(registration, change):
    source, _, current, state, client = registration
    if change == "extra-file":
        (source / "extra").write_text("unexpected")
    elif change == "payload":
        (source / "image_key").write_bytes(b"x" * 32)
    elif change == "fragment":
        (source / "resource-policy-fragment.rego").write_text("allow := true")
        update_manifest(source)
    else:
        (source / "image-security-policy.json").write_text('{"default":[{"type":"insecureAcceptAnything"}]}')
        update_manifest(source)
    with pytest.raises(ValueError):
        RECEIVER.install_release(source, current, state, client)
    assert not client.events


def test_missing_current_policy_never_creates_default_policy_or_uploads(registration):
    source, _, current, state, client = registration
    current.unlink()
    with pytest.raises(ValueError, match="CURRENT_RESOURCE_POLICY_FILE"):
        RECEIVER.install_release(source, current, state, client)
    assert not current.exists() and not client.events


def test_client_never_logs_rejected_secret_or_requires_local_jwt_claims(monkeypatch):
    client = RECEIVER.TrusteeClient("kbs-client", "https://trustee.example.com", "ca.pem", "opaque-credential")
    result = Mock(returncode=1, stdout=b"sensitive resource", stderr=b"opaque credential")
    run = Mock(return_value=result)
    monkeypatch.setattr(RECEIVER.subprocess, "run", run)
    with pytest.raises(ValueError) as error:
        client.set_resource("default/image-key/demo", "private-image-key")
    assert "sensitive resource" not in str(error.value) and "opaque credential" not in str(error.value)
    assert run.call_args.kwargs["capture_output"] is True
    assert run.call_args.args[0][-4:] == ["--path", "default/image-key/demo", "--resource-file", "private-image-key"]


def test_ambiguous_subprocess_failure_is_reported_without_response(monkeypatch):
    client = RECEIVER.TrusteeClient("kbs-client", "https://trustee.example.com", "ca.pem", "opaque-credential")
    run = Mock(side_effect=subprocess.TimeoutExpired(["kbs-client"], 90, output=b"private response"))
    monkeypatch.setattr(RECEIVER.subprocess, "run", run)
    with pytest.raises(ValueError) as error:
        client.set_policy("candidate.rego")
    assert "private response" not in str(error.value)
