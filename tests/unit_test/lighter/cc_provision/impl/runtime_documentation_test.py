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

"""Check that the ready-cluster guide documents every supported target."""

from pathlib import Path

from nvflare.lighter.cc_provision.kata_runtime_profile import RUNTIME_TARGETS, runtime_target

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"


def test_runtime_table_documents_every_supported_target():
    guide = (ROOT / "README.md").read_text()
    for name in RUNTIME_TARGETS:
        target = runtime_target(name)
        row = next(line for line in guide.splitlines() if line.startswith("|") and f"`{name}`" in line)
        cpu = "intel_tdx" if target["cpu_tee"] == "tdx" else "amd_sev_snp"
        gpu = "nvidia_cc" if target["gpu"] == "nvidia" else "none"
        assert f"`{cpu}`" in row
        assert f"`{gpu}`" in row
        assert ("nvidia.com/pgpu" in row) == bool(target["gpu_count"])
