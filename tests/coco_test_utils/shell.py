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


"""Shared factories for CoCo unit and integration contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples/devops/coco"


def require_coco_bash():
    """Check shell prerequisites without skipping the Python/static tests."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("CoCo shell tests require bash on PATH")
    # Lowercase expansion and fractional read timeouts need Bash 4; empty
    # arrays under nounset also require the Bash 4.4 behavior used by the harness.
    probe = (
        "set -eu\n"
        "printf '%s\\n' \"$BASH_VERSION\"\n"
        "value=ABC; [[ ${value,,} == abc ]]\n"
        'items=(); for item in "${items[@]}"; do :; done\n'
        'read -r -t 0.05 item <<< ready; [[ "$item" == ready ]]\n'
    )
    try:
        result = subprocess.run([bash, "-c", probe], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        pytest.skip("CoCo shell prerequisite probe timed out")
    if result.returncode:
        pytest.skip(
            "CoCo shell tests require lowercase expansion, fractional read timeouts and nounset-safe empty arrays "
            f"(Bash 4.4+; macOS system Bash 3.2 is insufficient): {result.stdout.strip()} {result.stderr.strip()}"
        )
    return bash
