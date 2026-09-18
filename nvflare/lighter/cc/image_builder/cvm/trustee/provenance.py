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

"""Record a clean CoCo Trustee v0.22.0 source revision and its built binary."""

import subprocess

from ..common.errors import require
from ..common.io import digest_file
from ..common.versions import TRUSTEE_COMMIT


def provenance(source, binary):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(commit == TRUSTEE_COMMIT, "Use the CoCo Trustee v0.22.0 source revision")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    require(not status, "Trustee source must be an unmodified upstream checkout")
    return {"trustee_commit": commit, "source_clean": True, "binary_sha256": digest_file(binary)}
