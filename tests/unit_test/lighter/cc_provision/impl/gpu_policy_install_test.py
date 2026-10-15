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

"""Run the stage 09 installer against isolated policy files and command stubs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SERVICE = ROOT / "examples/devops/coco/service"
PINNED = Path(__file__).parent / "fixtures/ear_default_policy_gpu.rego"

COMMON_STUB = r"""
KBS_CLIENT="$TEST_KBS_CLIENT"
TRUSTEE_PUBLIC_CERT="$TEST_CERT"
ADMIN_TOKEN="$TEST_ADMIN_TOKEN"
KBS_POLICY_DIR="$TEST_KBS_POLICY_DIR"
AS_STORAGE_DIR="$TEST_AS_STORAGE_DIR"
KBS_URL="https://test.invalid"
die() { printf '%s\n' "$*" >&2; exit 1; }
require_sudo() { :; }
need_file() { [[ -f "$1" ]] || die "missing file: $1"; }
lock_platform_reference_update() { :; }
configured_platform_values() { cat "$TEST_PLATFORM_VALUES"; }
install_platform_references() { printf 'references\n' >> "$TEST_EVENTS"; }
kbs_admin() {
    local id='' file=''
    while (( $# )); do
        case "$1" in
            --id) id="$2"; shift 2 ;;
            --policy-file) file="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    [[ -n "$id" && -f "$file" ]] || die 'invalid policy call'
    printf '%s\n' "$id" >> "$TEST_EVENTS"
    if [[ "$id" == default_gpu && "$TEST_BAD_GPU_READBACK" == 1 ]]; then
        printf 'unexpected persisted policy\n' > "$AS_STORAGE_DIR/attestation_service_policy/$id.rego"
    else
        cp "$file" "$AS_STORAGE_DIR/attestation_service_policy/$id.rego"
    fi
}
stat() {
    if [[ "$1" == -c && "$2" == %a && "$3" == "$ADMIN_TOKEN" ]]; then
        printf '600\n'
    else
        command stat "$@"
    fi
}
curl() { printf 'health\n' >> "$TEST_EVENTS"; }
"""


@pytest.fixture
def installation(tmp_path):
    service = tmp_path / "service"
    (service / "lib").mkdir(parents=True)
    (service / "policies").mkdir()
    (service / "10-verify-platform-reference-values.sh").write_text(
        '#!/usr/bin/env bash\nprintf "verified\\n" >> "$TEST_EVENTS"\n'
    )
    for relative in (
        "09-install-platform-policy.sh",
        "policies/default_cpu.rego",
        "policies/default_gpu.rego",
        "lib/platform-reference-values.py",
    ):
        shutil.copyfile(SERVICE / relative, service / relative)
    shutil.copyfile(
        ROOT / "examples/devops/coco/shared/platform-reference-values.py",
        service / "lib/platform-reference-schema.py",
    )
    (service / "lib/common.sh").write_text(COMMON_STUB)

    as_storage = tmp_path / "as"
    gpu_path = as_storage / "attestation_service_policy/default_gpu.rego"
    gpu_path.parent.mkdir(parents=True)
    resource = tmp_path / "kbs/resource-policy.rego"
    resource.parent.mkdir()
    resource.write_bytes(b"resource policy sentinel\n")
    for name in ("kbs-client", "cert", "admin-token"):
        (tmp_path / name).touch()
    (tmp_path / "admin-token").chmod(0o600)
    values = tmp_path / "values.json"
    values.write_text(
        json.dumps(
            {
                "snp_launch_measurement": "a" * 96,
                "snp_min_reported_tcb_bootloader": 1,
                "snp_min_reported_tcb_tee": 1,
                "snp_min_reported_tcb_snp": 1,
                "snp_min_reported_tcb_microcode": 1,
            }
        )
    )
    events = tmp_path / "events"
    events.touch()
    env = os.environ.copy()
    env.update(
        TEST_KBS_CLIENT=str(tmp_path / "kbs-client"),
        TEST_CERT=str(tmp_path / "cert"),
        TEST_ADMIN_TOKEN=str(tmp_path / "admin-token"),
        TEST_KBS_POLICY_DIR=str(resource.parent),
        TEST_AS_STORAGE_DIR=str(as_storage),
        TEST_PLATFORM_VALUES=str(values),
        TEST_EVENTS=str(events),
    )

    def run(active_gpu=None, bad_gpu_readback=False):
        if active_gpu is not None:
            gpu_path.write_bytes(active_gpu)
        result = subprocess.run(
            ["bash", str(service / "09-install-platform-policy.sh"), "--approve-pinned-platform"],
            env={**env, "TEST_BAD_GPU_READBACK": "1" if bad_gpu_readback else "0"},
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert resource.read_bytes() == b"resource policy sentinel\n"
        return result, events.read_text().splitlines(), gpu_path

    return run


def pinned_policy_bytes():
    # The fixture omits one upstream trailing space for git diff --check.
    return PINNED.read_bytes().replace(b"\n}\n\nelse := 3 if {", b"\n} \n\nelse := 3 if {")


def test_pinned_gpu_policy_upgrades_to_reviewed_policy(installation):
    result, events, gpu_path = installation(pinned_policy_bytes())
    assert result.returncode == 0, result.stderr
    assert gpu_path.read_bytes() == (SERVICE / "policies/default_gpu.rego").read_bytes()
    assert events == ["default_cpu", "default_gpu", "references", "verified", "health"]


def test_reviewed_gpu_policy_is_idempotent(installation):
    reviewed = (SERVICE / "policies/default_gpu.rego").read_bytes()
    first, _, _ = installation(reviewed)
    second, events, gpu_path = installation()
    assert first.returncode == second.returncode == 0
    assert gpu_path.read_bytes() == reviewed
    assert events == ["default_cpu", "default_gpu", "references", "verified", "health"] * 2


def test_unknown_gpu_policy_is_rejected_before_attestation_policy_writes(installation):
    unknown = b"package policy\ntrust_claims := {}\n"
    result, events, gpu_path = installation(unknown)
    assert result.returncode != 0
    assert "neither pinned Trustee" in result.stderr
    assert events == []
    assert gpu_path.read_bytes() == unknown


def test_gpu_readback_mismatch_fails_after_installation(installation):
    result, events, gpu_path = installation(pinned_policy_bytes(), bad_gpu_readback=True)
    assert result.returncode != 0
    assert "persisted GPU policy differs" in result.stderr
    assert events == ["default_cpu", "default_gpu", "references"]
    assert gpu_path.read_bytes() != (SERVICE / "policies/default_gpu.rego").read_bytes()
