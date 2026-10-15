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

"""GPU reference approval and staged replacement without network access."""

import importlib.util
import json
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
COCO = ROOT / "examples/devops/coco"
SOURCE = COCO / "service/lib/gpu-reference-values.py"
SPEC = importlib.util.spec_from_file_location("gpu_reference_values", SOURCE)
GPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU)


def values():
    return {
        "schema": GPU.SCHEMA,
        "allowed_driver_versions": ["550.54.15", "555.42.02"],
        "allowed_vbios_versions": ["96.00.5E.00.01"],
    }


def test_exact_gpu_reference_schema_and_staged_payloads():
    approved = values()
    assert GPU.validate_values(approved) == approved
    assert GPU.stage_payload(approved, "clear") == {"allowed_driver_versions": []}
    assert GPU.stage_payload(approved, "vbios") == {"allowed_vbios_versions": ["96.00.5E.00.01"]}
    assert GPU.stage_payload(approved, "drivers") == {
        "allowed_driver_versions": sorted(approved["allowed_driver_versions"])
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.pop("allowed_driver_versions"),
        lambda v: v.update(extra=[]),
        lambda v: v.update(schema="wrong"),
        lambda v: v.update(allowed_driver_versions=[]),
        lambda v: v.update(allowed_driver_versions=["550.54.15", "550.54.15"]),
        lambda v: v.update(allowed_vbios_versions=["with space"]),
        lambda v: v.update(allowed_vbios_versions=[True]),
        lambda v: v.update(allowed_vbios_versions=["a" * 129]),
    ],
)
def test_invalid_gpu_approvals_fail_before_install(mutation):
    approved = values()
    mutation(approved)
    with pytest.raises(ValueError):
        GPU.validate_values(approved)


def test_duplicate_json_key_and_oversize_file_fail(tmp_path):
    path = tmp_path / "gpu.json"
    path.write_text('{"schema":"coco-gpu-reference-values/v1","schema":"duplicate"}')
    with pytest.raises(ValueError, match="Duplicate"):
        GPU.load_values(path)
    path.write_bytes(b"x" * (GPU.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        GPU.load_values(path)


@pytest.mark.parametrize("stage", ["clear", "vbios", "drivers"])
def test_readback_requires_exact_list(stage):
    approved = values()
    name, expected = GPU.expected_stage(approved, stage)
    assert name in GPU.IDS
    GPU.compare_stage(approved, stage, json.dumps(expected))
    GPU.compare_stage(approved, stage, json.dumps(json.dumps(expected)))
    for wrong in (None, {}, "wrong", expected + ["extra"]):
        with pytest.raises(ValueError):
            GPU.compare_stage(approved, stage, json.dumps(wrong))


def test_gpu_snapshot_is_private_and_cpu_selector_is_preserved(tmp_path):
    env = tmp_path / "platform.env"
    env.write_text('PLATFORM_REFERENCE_VALUES_FILE="/existing/cpu.json"\n')
    GPU.update_env(values(), env)
    snapshot = tmp_path / "approved-gpu-reference-values.json"
    assert GPU.load_values(snapshot) == values()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert 'PLATFORM_REFERENCE_VALUES_FILE="/existing/cpu.json"' in env.read_text()
    assert "GPU_REFERENCE_VALUES_FILE=" + shlex.quote(str(snapshot)) in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    snapshot.unlink()
    snapshot.symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="symlink"):
        GPU.update_env(values(), env)


def test_maximal_valid_compact_input_remains_readable_after_snapshot(tmp_path):
    approved = {
        "schema": GPU.SCHEMA,
        "allowed_driver_versions": [f"D{i:02d}" + "a" * 119 for i in range(64)],
        "allowed_vbios_versions": [f"V{i:02d}" + "b" * 119 for i in range(64)],
    }
    assert len(json.dumps(approved, separators=(",", ":")).encode()) < GPU.MAX_BYTES
    assert len(json.dumps(approved, indent=2).encode()) > GPU.MAX_BYTES
    env = tmp_path / "platform.env"
    env.write_text('GPU_REFERENCE_VALUES_FILE=""\n')
    GPU.update_env(approved, env)
    snapshot = tmp_path / "approved-gpu-reference-values.json"
    assert snapshot.stat().st_size <= GPU.MAX_BYTES
    assert GPU.load_values(snapshot) == approved

    # An exactly maximal compact input is valid; snapshotting must not add a byte.
    approved["allowed_driver_versions"] = [
        version + ("x" * 6 if index < 48 else "") for index, version in enumerate(approved["allowed_driver_versions"])
    ]
    assert len(json.dumps(approved, separators=(",", ":")).encode()) == GPU.MAX_BYTES
    GPU.update_env(approved, env)
    assert snapshot.stat().st_size == GPU.MAX_BYTES
    assert GPU.load_values(snapshot) == approved


def test_gpu_post_stage_touches_only_selected_reference(monkeypatch):
    sent = []
    monkeypatch.setattr(GPU._transport, "post_reference", lambda message, *_: sent.append(message))
    approved = values()
    for stage in ("clear", "vbios", "drivers"):
        GPU.post_stage(approved, stage, "https://service.test", "ca", "token")
    assert len(sent) == 3
    import base64

    decoded = [json.loads(base64.b64decode(message["payload"])) for message in sent]
    assert decoded == [
        {"allowed_driver_versions": []},
        {"allowed_vbios_versions": approved["allowed_vbios_versions"]},
        {"allowed_driver_versions": sorted(approved["allowed_driver_versions"])},
    ]
    assert all(not set(item) & set(GPU._transport.FIELDS) for item in decoded)


def test_gpu_release_gate_precedes_publication_and_cpu_bypasses_it(tmp_path):
    script = (COCO / "service/12-install-trusted-service-handoff.sh").read_text()
    start = script.index('lock_platform_reference_update\nif [[ "$RELEASE_GPU" == nvidia ]]; then')
    end = script.index("if grep -Fq", start)
    gate = script[start:end]
    assert script.index('authorization["gpu"]') < start
    assert script.index('ACTIVE_POLICY="${TRUSTEE_ROOT}') < start
    for gpu, ready in (("nvidia", False), ("nvidia", True), ("none", False)):
        program = (
            "set -Eeuo pipefail\n"
            f"RELEASE_GPU={shlex.quote(gpu)}\n"
            "lock_platform_reference_update() { echo LOCKED; }\n"
            f"verify_gpu_readiness() {{ echo CHECKED; return {0 if ready else 1}; }}\n" + gate + "echo WOULD_PUBLISH\n"
        )
        result = subprocess.run(["bash", "-c", program], capture_output=True, text=True, timeout=5)
        assert ("WOULD_PUBLISH" in result.stdout) == (ready or gpu == "none")
        assert ("CHECKED" in result.stdout) == (gpu == "nvidia")
        assert result.stdout.startswith("LOCKED\n")


def test_gpu_reference_scripts_are_packaged_as_tracked_role_files():
    source = COCO / "role_kits.py"
    spec = importlib.util.spec_from_file_location("coco_role_kits_gpu", source)
    role_kits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(role_kits)
    names = role_kits.package_files(COCO)
    for name in (
        "service/lib/gpu-reference-values.py",
        "service/policies/default_gpu.rego",
        "service/02g-install-gpu-reference-values.sh",
        "service/10g-verify-gpu-reference-values.sh",
        "service/GPU-REFERENCE-VALUES.md",
    ):
        assert name in names
