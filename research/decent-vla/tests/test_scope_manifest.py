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

"""scope_manifest.json — provenance for the two scope axes of a run.

Written at run start next to partition_manifest.json: the configured
trainable/shared patterns, the RESOLVED parameter names, and the counts/bytes
that report the parameter and uplink sizes directly.
"""
from __future__ import annotations

import json

import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.configs.schemas import BackendConfig, FLConfig
from decent_vla.runtime.scope_manifest import write_scope_manifest


def _backend(**kw):
    cfg = BackendConfig(name="mock", extra={"model_seed": 0}, **kw)
    return MockBackend.from_config(cfg, torch.device("cpu")), cfg


def test_manifest_records_patterns_names_and_counts(tmp_path):
    backend, bcfg = _backend(shared_params=["fc2.*"])
    fl = FLConfig()
    p = write_scope_manifest(tmp_path, bcfg, fl, backend)
    blob = json.loads(p.read_text())
    assert blob["shared_params"] == ["fc2.*"]
    assert blob["trainable_params"] is None
    assert set(blob["shared_names"]) == {"fc2.weight", "fc2.bias"}
    assert set(blob["personal_names"]) == {"fc1.weight", "fc1.bias"}
    fc2_numel = 32 * 7 + 7
    assert blob["counts"]["shared_numel"] == fc2_numel
    assert blob["counts"]["trainable_numel"] > fc2_numel
    # Uplink pricing: fp32 bytes of the shared subset + the actual wire bytes.
    assert blob["bytes"]["shared_fp32"] == 4 * fc2_numel
    assert blob["wire_dtype"] == "fp16"
    assert blob["bytes"]["shared_wire"] == 2 * fc2_numel


def test_manifest_no_patterns_everything_shared(tmp_path):
    backend, bcfg = _backend()
    p = write_scope_manifest(tmp_path, bcfg, FLConfig(), backend)
    blob = json.loads(p.read_text())
    assert blob["personal_names"] == []
    assert blob["counts"]["shared_numel"] == blob["counts"]["trainable_numel"]


def test_manifest_prices_the_uplink_of_a_per_direction_wire_dtype(tmp_path):
    """A per-direction wire_dtype is a mapping; the manifest prices the delta (uplink) direction."""
    backend, bcfg = _backend(shared_params=["fc2.*"])
    fl = FLConfig(wire_dtype={"broadcast": "fp32", "delta": "fp16"})
    blob = json.loads(write_scope_manifest(tmp_path, bcfg, fl, backend).read_text())
    assert blob["bytes"]["shared_wire"] == 2 * (32 * 7 + 7)
