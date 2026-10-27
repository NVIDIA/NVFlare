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

"""Reference updates use the existing Trustee API without rewriting installation state."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COCO = Path(__file__).resolve().parents[5] / "examples/devops/coco"
STAGE = "02-install-platform-reference-values.sh"
FIELDS = ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam", "tdvfkernel", "tdvfkernelparams")
STUB_COMMON = r"""
source "$SCRIPT_DIR/platform.env"
die() { printf '%s\n' "$*" >&2; exit 1; }
lock_platform_reference_update() { printf 'lock references\n' >>"$TEST_CALLS"; }
verify_platform_references() {
    while IFS= read -r key; do
        raw=$("$KBS_CLIENT" --id "$key")
        python3 "$SCRIPT_DIR/lib/platform-reference-values.py" compare-reference "$1" "$key" "$raw"
    done < <(python3 "$SCRIPT_DIR/lib/platform-reference-values.py" reference-ids "$1")
}
"""


def reference_values(*missing):
    profile = {"id": "approved-tdx"}
    profile.update({key: "a" * (16 if key == "xfam" else 96) for key in FIELDS if key not in missing})
    return {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": [profile]}


@pytest.fixture
def service(tmp_path):
    role = tmp_path / "service"
    library = role / "lib"
    library.mkdir(parents=True)
    shutil.copyfile(COCO / "service" / STAGE, role / STAGE)
    shutil.copyfile(COCO / "service/lib/platform-reference-values.py", library / "platform-reference-values.py")
    shutil.copyfile(COCO / "shared/platform-reference-values.py", library / "platform-reference-schema.py")
    (library / "common.sh").write_text(STUB_COMMON)
    old_local = role / "approved-platform-reference-values.json"
    old_local.write_text(json.dumps(reference_values("rtmr_0", "rtmr_3")))
    old_local.chmod(0o600)
    state = tmp_path / "live-references.json"
    state.write_text(
        json.dumps(
            {
                "snp_launch_measurement": ["b" * 96],
                "snp_min_reported_tcb_bootloader": 1,
                "coco_tdx_profiles_v2": reference_values("rtmr_0", "rtmr_3")["profiles"],
            }
        )
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    calls = tmp_path / "calls"
    updates = tmp_path / "updates"
    tools = tmp_path / "bin"
    tools.mkdir()
    client = tools / "kbs-client"
    client.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "state=json.loads(Path(os.environ['TEST_REFERENCE_STATE']).read_text())\n"
        "print(json.dumps(state[sys.argv[sys.argv.index('--id')+1]]))\n"
    )
    client.chmod(0o700)
    proxy = tools / "python3"
    proxy.write_text(
        f"#!{sys.executable}\n"
        + r"""
import base64, importlib.util, json, os, sys
from pathlib import Path
if len(sys.argv) > 2 and sys.argv[2] == "install-references":
    spec = importlib.util.spec_from_file_location("live_reference_installer", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def post(message, *_):
        path = Path(os.environ["TEST_REFERENCE_STATE"])
        state = json.loads(path.read_text())
        payload = json.loads(base64.b64decode(message["payload"]))
        state.update(payload)
        path.write_text(json.dumps(state))
        with Path(os.environ["TEST_REFERENCE_UPDATES"]).open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
    module.post_reference = post
    sys.argv = sys.argv[1:]
    module.main()
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""
    )
    proxy.chmod(0o700)
    (role / "platform.env").write_text(
        f"STATE_DIR={state_dir}\nKBS_URL=https://existing-trustee.example\n"
        f"TRUSTEE_PUBLIC_CERT=/existing/ca.crt\nADMIN_TOKEN=/existing/admin-token\nKBS_CLIENT={client}\n"
    )
    env = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEST_CALLS": str(calls),
        "TEST_REFERENCE_STATE": str(state),
        "TEST_REFERENCE_UPDATES": str(updates),
    }
    return role, env, state, updates


def run_update(role, env, values, mode=None):
    arguments = ["bash", str(role / STAGE), str(values)]
    if mode is not None:
        arguments.append(mode)
    return subprocess.run(arguments, env=env, capture_output=True, text=True, timeout=10)


def test_reference_update_preserves_other_tee_and_local_installation_configuration(tmp_path, service):
    role, env, state, updates = service
    old_env = (role / "platform.env").read_bytes()
    old_snapshot = (role / "approved-platform-reference-values.json").read_bytes()
    previous = json.loads(state.read_text())
    values = tmp_path / "reviewed-values.json"
    values.write_text(json.dumps(reference_values()))
    result = run_update(role, env, values)
    assert result.returncode == 0, result.stderr
    current = json.loads(state.read_text())
    assert current["coco_tdx_profiles_v2"] == reference_values()["profiles"]
    assert current["snp_launch_measurement"] == previous["snp_launch_measurement"]
    assert current["snp_min_reported_tcb_bootloader"] == previous["snp_min_reported_tcb_bootloader"]
    assert len(updates.read_text().splitlines()) == 1
    assert (role / "platform.env").read_bytes() == old_env
    assert (role / "approved-platform-reference-values.json").read_bytes() == old_snapshot


@pytest.mark.parametrize("missing", [("rtmr_0",), ("rtmr_3",), ("rtmr_0", "rtmr_3")])
def test_reference_update_rejects_incomplete_new_values_before_api_writes(tmp_path, service, missing):
    role, env, state, updates = service
    before = state.read_bytes()
    values = tmp_path / "incomplete-values.json"
    values.write_text(json.dumps(reference_values(*missing)))
    result = run_update(role, env, values)
    assert result.returncode != 0
    assert "all eight measurement fields" in result.stderr
    assert not updates.exists()
    assert state.read_bytes() == before


def test_verify_only_reads_existing_references_without_api_writes(tmp_path, service):
    role, env, state, updates = service
    current = json.loads(state.read_text())
    current["coco_tdx_profiles_v2"] = reference_values()["profiles"]
    state.write_text(json.dumps(current))
    before = state.read_bytes()
    values = tmp_path / "reviewed-values.json"
    values.write_text(json.dumps(reference_values()))
    result = run_update(role, env, values, "--verify-only")
    assert result.returncode == 0, result.stderr
    assert state.read_bytes() == before
    assert not updates.exists()


@pytest.mark.parametrize(
    "script, option",
    [
        ("02-install-platform-reference-values.sh", "--approve-platform-reference-values"),
        ("02g-install-gpu-reference-values.sh", "--approve-gpu-reference-values"),
    ],
)
def test_removed_approval_flags_fail_before_reading_values_or_service_configuration(tmp_path, script, option):
    role = tmp_path / "service"
    role.mkdir()
    shutil.copyfile(COCO / "service" / script, role / script)
    result = subprocess.run(
        ["bash", str(role / script), str(tmp_path / "missing-values.json"), option],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert "No such file" not in result.stderr
    assert sorted(path.name for path in role.iterdir()) == [script]
