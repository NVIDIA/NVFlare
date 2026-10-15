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

"""Run the GPU reference shell stages against an isolated fake KBS/RVPS."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SERVICE = ROOT / "examples/devops/coco/service"
POST = r"""
import json, os, sys
from pathlib import Path

stage = sys.argv[1]
root = Path(os.environ["TEST_ROOT"])
values = json.loads((root / "approved.json").read_text())
state = root / "state"
calls = root / "calls"
history = calls.read_text().splitlines() if calls.exists() else []
history.append(stage)
calls.write_text("\n".join(history) + "\n")
mode = os.environ.get("TEST_MODE", "normal")
if stage == "clear":
    if mode == "failed_compensation" and history.count("clear") > 1:
        raise SystemExit(17)
    (state / "allowed_driver_versions.json").write_text("[]")
elif stage == "vbios":
    (state / "allowed_vbios_versions.json").write_text(json.dumps(values["allowed_vbios_versions"]))
elif stage == "drivers":
    (state / "allowed_driver_versions.json").write_text(json.dumps(sorted(values["allowed_driver_versions"])))
    if mode in ("ambiguous_final_write", "failed_compensation"):
        raise SystemExit(18)
    if mode == "interrupt":
        import signal
        os.kill(os.getppid(), signal.SIGTERM)
"""
COMMON = r"""
source "$TEST_ROOT/platform.env"
SCRIPT_ROOT="$TEST_ROOT"
KBS_CLIENT="$TEST_ROOT/kbs-client"
ADMIN_TOKEN="$TEST_ROOT/admin-token"
TRUSTEE_PUBLIC_CERT="$TEST_ROOT/cert"
KBS_POLICY_DIR="$TEST_ROOT/kbs-policy"
AS_STORAGE_DIR="$TEST_ROOT/as"
TRUSTEE_PROJECT="test"
TRUSTEE_COMPOSE="$TEST_ROOT/compose.yml"
KBS_URL="https://service.test:8443"
lock_platform_reference_update() { :; }
require_sudo() { :; }
sleep() { :; }
need_file() { [[ -f "$1" ]]; }
die() { echo "$*" >&2; exit 1; }
python3() {
    if [[ "${2:-}" == post-stage ]]; then
        "$REAL_PYTHON" "$TEST_ROOT/fake-post.py" "$4"
    else
        "$REAL_PYTHON" "$@"
    fi
}
sudo() {
    if [[ "$1" == docker ]]; then
        echo RESTART >> "$TEST_ROOT/calls"
        if [[ ${TEST_MODE:-} == lost_on_restart ]]; then
            rm -f "$TEST_ROOT/state/allowed_driver_versions.json"
        fi
        return
    fi
    "$@"
}
kbs_admin() {
    [[ "$1" == get-reference-value && "$2" == --id ]] || return 2
    cat "$TEST_ROOT/state/$3.json"
}
"""


@pytest.fixture
def kit(tmp_path):
    for name in ("02g-install-gpu-reference-values.sh", "10g-verify-gpu-reference-values.sh"):
        shutil.copyfile(SERVICE / name, tmp_path / name)
    for name in ("gpu-reference-values.py", "platform-reference-values.py"):
        destination = tmp_path / "lib" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(SERVICE / "lib" / name, destination)
    shutil.copyfile(
        SERVICE.parent / "shared/platform-reference-values.py", tmp_path / "lib/platform-reference-schema.py"
    )
    (tmp_path / "lib/common.sh").write_text(COMMON)
    (tmp_path / "fake-post.py").write_text(POST)
    (tmp_path / "policies").mkdir()
    shutil.copyfile(SERVICE / "policies/default_gpu.rego", tmp_path / "policies/default_gpu.rego")
    (tmp_path / "as/attestation_service_policy").mkdir(parents=True)
    shutil.copyfile(SERVICE / "policies/default_gpu.rego", tmp_path / "as/attestation_service_policy/default_gpu.rego")
    (tmp_path / "as/attestation_service_policy/default_cpu.rego").write_text("approved cpu policy\n")
    (tmp_path / "kbs-policy").mkdir()
    (tmp_path / "kbs-policy/resource-policy.rego").write_text("default deny\n")
    for name in ("kbs-client", "admin-token", "cert", "compose.yml"):
        (tmp_path / name).touch()
    (tmp_path / "platform.env").write_text(
        'PLATFORM_REFERENCE_VALUES_FILE="/existing/cpu-reference-values.json"\nGPU_REFERENCE_VALUES_FILE=""\n'
    )
    approved = {
        "schema": "coco-gpu-reference-values/v1",
        "allowed_driver_versions": ["550.54.15"],
        "allowed_vbios_versions": ["96.00.5E.00.01"],
    }
    (tmp_path / "approved.json").write_text(json.dumps(approved))
    (tmp_path / "state").mkdir()
    (tmp_path / "state/allowed_driver_versions.json").write_text('["old-driver"]')
    (tmp_path / "state/allowed_vbios_versions.json").write_text('["old-vbios"]')
    # The production kit runs on Linux, whose realpath supports -e.
    # macOS's realpath lacks that option, so provide its strict-path equivalent.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    realpath = bin_dir / "realpath"
    realpath.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        'assert sys.argv[1] == "-e"\n'
        "print(pathlib.Path(sys.argv[-1]).resolve(strict=True))\n"
    )
    realpath.chmod(0o755)
    environment = {
        **os.environ,
        "TEST_ROOT": str(tmp_path),
        "REAL_PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    return tmp_path, environment


def run_stage(kit, mode="normal"):
    root, environment = kit
    return subprocess.run(
        [
            "bash",
            str(root / "02g-install-gpu-reference-values.sh"),
            str(root / "approved.json"),
            "--approve-gpu-reference-values",
        ],
        env={**environment, "TEST_MODE": mode},
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_staged_install_preserves_cpu_and_policies(kit):
    root, _ = kit
    before = {
        name: (root / name).read_bytes()
        for name in (
            "as/attestation_service_policy/default_cpu.rego",
            "as/attestation_service_policy/default_gpu.rego",
            "kbs-policy/resource-policy.rego",
        )
    }
    result = run_stage(kit)
    assert result.returncode == 0, result.stderr
    assert (root / "calls").read_text().splitlines() == ["clear", "vbios", "drivers"]
    assert json.loads((root / "state/allowed_driver_versions.json").read_text()) == ["550.54.15"]
    assert json.loads((root / "state/allowed_vbios_versions.json").read_text()) == ["96.00.5E.00.01"]
    assert 'PLATFORM_REFERENCE_VALUES_FILE="/existing/cpu-reference-values.json"' in (root / "platform.env").read_text()
    assert all((root / name).read_bytes() == original for name, original in before.items())


@pytest.mark.parametrize("mode", ["ambiguous_final_write", "failed_compensation", "interrupt"])
def test_ambiguous_or_interrupted_final_write_compensates_or_reports_unknown(kit, mode):
    root, _ = kit
    result = run_stage(kit, mode)
    assert result.returncode != 0
    assert (root / "calls").read_text().splitlines() == ["clear", "vbios", "drivers", "clear"]
    drivers = json.loads((root / "state/allowed_driver_versions.json").read_text())
    if mode == "failed_compensation":
        assert drivers == ["550.54.15"]
        assert "UNKNOWN" in result.stderr
    else:
        assert drivers == []
        assert "cleared after failure" in result.stderr


@pytest.mark.parametrize("mode", ["normal", "lost_on_restart"])
def test_restart_readback_requires_persisted_gpu_references(kit, mode):
    root, environment = kit
    assert run_stage(kit).returncode == 0
    result = subprocess.run(
        ["bash", str(root / "10g-verify-gpu-reference-values.sh"), str(root / "approved.json"), "--restart-rvps"],
        env={**environment, "TEST_MODE": mode},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert ("persistence verified" in result.stdout) == (mode == "normal")
    assert (result.returncode == 0) == (mode == "normal")
