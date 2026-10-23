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

"""Offline checks for executable examples and synchronized runtime diagrams."""

import html
import re
import shlex
from pathlib import Path

from nvflare.lighter.cc_provision.kata_runtime_profile import RUNTIME_TARGETS, runtime_target

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"


def test_runtime_table_documents_every_supported_target():
    guide = (ROOT / "RUNTIME-VARIANTS.md").read_text()
    for name in RUNTIME_TARGETS:
        target = runtime_target(name)
        row = next(line for line in guide.splitlines() if line.startswith("|") and f"`{name}`" in line)
        cpu = "intel_tdx" if target["cpu_tee"] == "tdx" else "amd_sev_snp"
        assert f"`{cpu}`" in row
        assert f"`{target['gpu']}`" in row
        assert ("nvidia.com/pgpu" in row) == bool(target["gpu_count"])


def test_documented_release_checks_select_target_from_authorization():
    commands = []
    for path in ROOT.rglob("*.md"):
        for block in re.findall(r"```(?:bash|sh)\n(.*?)```", path.read_text(), re.S):
            for line in block.replace("\\\n", " ").splitlines():
                if not re.match(r"\s*(?:bash\s+)?\./13-verify-workload-release\.sh\s", line):
                    continue
                args = shlex.split(line, comments=True)
                if args and args[0] == "bash":
                    args = args[1:]
                if args and Path(args[0]).name == "13-verify-workload-release.sh":
                    commands.append((path, args))
    assert commands, "No workload-release verification example was checked"
    for path, args in commands:
        assert len(args) == 4, f"{path}: supply release, log window and reviewed authorization"
        assert args[3].endswith("/release-authorization.json"), str(path)


def test_sequence_html_embeds_the_current_mermaid_source():
    source = (ROOT / "docs/coco-four-party-sequence.mmd").read_text().strip()
    page = (ROOT / "docs/coco-four-party-sequence.html").read_text()
    embedded = re.search(r"<details>.*?<pre>(.*?)</pre>", page, re.S)
    assert embedded is not None
    assert html.unescape(embedded[1]).strip() == source
