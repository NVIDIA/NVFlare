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


"""In-process CoCo policy and validation contracts."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

POLICY = ROOT / "examples/devops/coco/service/policies/default_gpu.rego"

PINNED = Path(__file__).parent / "fixtures/ear_default_policy_gpu.rego"

PINNED_BLOB_SHA = "0cd04bb6e236653a3d633bcefa1da8c3e42eabfd"

ZERO_PADDING = ("file-system", "instance-identity", "runtime-opaque", "storage-opaque", "sourced-data")


def test_gpu_policy_is_pinned_trustee_policy_with_only_padding_claims_removed():
    pinned = PINNED.read_bytes().replace(b"\n}\n\nelse := 3 if {", b"\n} \n\nelse := 3 if {")
    blob = b"blob " + str(len(pinned)).encode() + b"\0" + pinned
    assert hashlib.sha1(blob).hexdigest() == PINNED_BLOB_SHA

    lines = pinned.decode().splitlines()
    removed = [line for line in lines if any(f'"{name}":' in line for name in ZERO_PADDING)]
    assert len(removed) == len(ZERO_PADDING)
    expected = [line for line in lines if line not in removed]

    def substantive(source):
        return [line.rstrip() for line in source if line.strip() and not line.lstrip().startswith("#")]

    assert substantive(POLICY.read_text().splitlines()) == substantive(expected)
