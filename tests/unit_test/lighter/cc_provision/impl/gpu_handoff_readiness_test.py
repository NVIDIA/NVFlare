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

"""Run the release installer through authentication, reconstruction and readiness."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import workload_release as release
from nvflare.lighter.cc_provision.workload_security import HANDOFF_FILES
from tests.unit_test.lighter.cc_provision.impl.deployment_guards_test import require_coco_bash

COCO = Path(__file__).resolve().parents[5] / "examples/devops/coco"
SERVICE = COCO / "service"
IMAGE = "secure-services.example.com:5000/workload@sha256:" + "c" * 64


def run_installer(tmp_path, runtime, ready, attack=None):
    """Stub host services, retaining the full shell script and actual handoff checks.

    The OPA boundary deliberately terminates the run after readiness, before
    interactive approval and publication. It proves CPU releases reach policy
    validation even when GPU readiness would fail, without emulating publication.
    """
    bash = require_coco_bash()
    kit = tmp_path / "kit"
    (kit / "lib").mkdir(parents=True)
    (kit / "policies").mkdir()
    script = kit / "12-install-trusted-service-handoff.sh"
    shutil.copyfile(SERVICE / script.name, script)
    for name in ("workload-release.py", "workload-security-context.py"):
        (kit / "lib" / name).symlink_to(SERVICE / "lib" / name)
    for name in ("workload-resource-policy.rego.template", "workload-resource-policy-v1.rego.template"):
        shutil.copyfile(SERVICE / "policies" / name, kit / "policies" / name)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    paths = [f"default/{kind}/demo" for kind in ("image-key", "sig-public-key", "security-policy")]
    auth = release.build_authorization(runtime, "demo", IMAGE, ["/bin/workload"], "ab" * 32, paths)
    fragment = release.render_fragment(auth, (SERVICE / "policies/workload-resource-policy.rego.template").read_text())
    if attack == "invalid-target":
        auth["gpu"] = "none"
    (incoming / "release-authorization.json").write_text(json.dumps(auth))
    (incoming / "resource-policy-fragment.rego").write_text(fragment)
    (incoming / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----\n")
    (incoming / "image_key").write_bytes(b"k" * 32)
    (incoming / "image_key").chmod(0o600)
    (incoming / "image-security-policy.json").write_text(
        json.dumps(
            {
                "default": [{"type": "reject"}],
                "transports": {
                    "docker": {
                        IMAGE.split("@")[0]: [
                            {
                                "type": "sigstoreSigned",
                                "keyPath": "kbs:///default/sig-public-key/demo",
                                "signedIdentity": {"type": "matchRepository"},
                            }
                        ]
                    }
                },
            }
        )
    )
    manifest = "".join(
        hashlib.sha256((incoming / name).read_bytes()).hexdigest() + "  " + name + "\n"
        for name in sorted(HANDOFF_FILES)
    )
    (incoming / "SHA256SUMS").write_text(manifest)
    pin = hashlib.sha256(manifest.encode()).hexdigest()
    if attack == "unauthenticated-target":
        auth["gpu"] = "none"
        (incoming / "release-authorization.json").write_text(json.dumps(auth))

    trustee = tmp_path / "trustee"
    policy = trustee / "kbs/data/kbs-policy/resource-policy.rego"
    policy.parent.mkdir(parents=True)
    initial_policy = "package policy\nimport rego.v1\ndefault allow := false\n"
    policy.write_text(initial_policy)
    storage = trustee / "kbs/data/kbs-storage"
    storage.mkdir()
    (trustee / "docker-compose.yml").write_text("services: {}\n")
    for name in ("cert", "token", "client"):
        (tmp_path / name).write_text("fixture\n")
    (tmp_path / "client").chmod(0o700)
    events = tmp_path / "events"
    (kit / "lib/common.sh").write_text(
        f"TRUSTEE_ROOT={shlex.quote(str(trustee))}\n"
        f"TRUSTEE_PUBLIC_CERT={shlex.quote(str(tmp_path / 'cert'))}\n"
        f"ADMIN_TOKEN={shlex.quote(str(tmp_path / 'token'))}\n"
        f"KBS_CLIENT={shlex.quote(str(tmp_path / 'client'))}\n"
        f"EVENTS={shlex.quote(str(events))}\n"
        "SERVICE_FQDN=secure-services.example.com\nREGISTRY_PORT=5000\nKBS_URL=https://kbs.invalid\n"
        'die() { echo "$*" >&2; exit 1; }\n'
        "need_cmd() { :; }\n"
        'need_file() { [[ -f "$1" ]] || die "missing $1"; }\n'
        f'python3() {{ {shlex.quote(sys.executable)} "$@"; }}\n'
        # Isolate the root ownership prerequisite without requiring root fixtures.
        'stat() { case "$2" in %a) echo 600 ;; %U:%G) echo root:root ;; *) return 1 ;; esac; }\n'
        + ('sha256sum() { shasum -a 256 "$@"; }\n' if sys.platform == "darwin" else "")
        + "openssl() { :; }\ncurl() { :; }\n"
        'lock_platform_reference_update() { echo LOCK >> "$EVENTS"; }\n'
        f'verify_gpu_readiness() {{ echo GPU_READINESS >> "$EVENTS"; return {0 if ready else 47}; }}\n'
        'opa() { echo POLICY_VALIDATION >> "$EVENTS"; return 48; }\n'
        'sudo() { echo UNEXPECTED_PUBLICATION >> "$EVENTS"; return 49; }\n'
    )
    result = subprocess.run(
        [bash, str(script), str(incoming), pin],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert policy.read_text() == initial_policy
    assert list(storage.iterdir()) == []
    assert not (tmp_path / "trustee-release-backups").exists()
    return result, events.read_text().splitlines() if events.exists() else []


@pytest.mark.parametrize("runtime", release.RUNTIME_TARGETS)
@pytest.mark.parametrize("ready", [False, True])
def test_full_release_installer_gates_validated_gpu_target_before_publication(tmp_path, runtime, ready):
    result, events = run_installer(tmp_path, runtime, ready)
    gpu = release.RUNTIME_TARGETS[runtime][1] == "nvidia"
    if gpu and not ready:
        assert result.returncode == 47, result.stderr
        assert events == ["LOCK", "GPU_READINESS"]
    else:
        assert result.returncode == 48, result.stderr
        assert events == ["LOCK"] + (["GPU_READINESS"] if gpu else []) + ["POLICY_VALIDATION"]


@pytest.mark.parametrize("attack", ["invalid-target", "unauthenticated-target"])
def test_release_installer_rejects_invalid_or_unauthenticated_gpu_metadata(tmp_path, attack):
    result, events = run_installer(tmp_path, "kata-qemu-nvidia-gpu-snp", False, attack)
    assert result.returncode != 0
    assert events == []
    assert (
        "incorrect required EAR submodule set" if attack == "invalid-target" else "payload digest mismatch"
    ) in result.stderr
