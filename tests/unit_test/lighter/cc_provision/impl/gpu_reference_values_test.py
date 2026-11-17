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


def test_maximal_valid_compact_gpu_input_loads_without_reformatting(tmp_path):
    approved = {
        "schema": GPU.SCHEMA,
        "allowed_driver_versions": [f"D{i:02d}" + "a" * 119 for i in range(64)],
        "allowed_vbios_versions": [f"V{i:02d}" + "b" * 119 for i in range(64)],
    }
    approved["allowed_driver_versions"] = [
        version + ("x" * 6 if index < 48 else "") for index, version in enumerate(approved["allowed_driver_versions"])
    ]
    compact = json.dumps(approved, separators=(",", ":"))
    assert len(compact.encode()) == GPU.MAX_BYTES
    path = tmp_path / "approved.json"
    path.write_text(compact)
    assert GPU.load_values(path) == approved
