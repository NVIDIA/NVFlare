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

"""Record an upstream Trustee service or the reviewed selector-capable client."""

import hashlib
import subprocess
from pathlib import Path

from ..common.errors import require
from ..common.io import digest_file
from ..common.versions import GUEST_COMPONENTS_SELECTOR_COMMIT, TRUSTEE_COMMIT

CLIENT_PATCH = Path(__file__).parents[1] / "build/kbs_client_policy_selector.patch"


def provenance(source, binary):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(commit == TRUSTEE_COMMIT, "Use the CoCo Trustee v0.22.0 source revision")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    result = {"trustee_commit": commit, "binary_sha256": digest_file(binary)}
    if not status:
        return dict(result, source_clean=True)
    diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary", "--no-ext-diff", "--no-color", "HEAD"]
    )
    expected = CLIENT_PATCH.read_bytes()
    require(
        status.splitlines() and all(not line.startswith("??") for line in status.splitlines()) and diff == expected,
        "Trustee source must be unmodified or contain only the reviewed kbs-client selector patch",
    )
    return dict(
        result,
        source_clean=False,
        kbs_client_patch_sha256=hashlib.sha256(expected).hexdigest(),
        guest_components_commit=GUEST_COMPONENTS_SELECTOR_COMMIT,
    )
