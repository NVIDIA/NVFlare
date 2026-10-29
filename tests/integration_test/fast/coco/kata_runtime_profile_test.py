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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import kata_runtime_profile
from tests.coco_test_utils.process import COMMAND_TIMEOUT, run_coco_command

ROOT = Path(__file__).resolve().parents[4] / "examples/devops/coco"


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "shared/kata-runtime-profile.py",
        ROOT / "trusted_system/lib/kata-runtime-profile.py",
        Path(kata_runtime_profile.__file__),
    ],
)
@pytest.mark.parametrize(
    "arguments", [[], ["target", "kata-qemu-snp", "--field", "cpu_tee"], ["enable", "unused.toml"]]
)
def test_removed_runtime_cli_rejects_direct_execution(path, arguments):
    result = run_coco_command(
        [sys.executable, str(path), *arguments], capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )
    assert result.returncode != 0
    assert not result.stdout
    assert "Kata runtime configuration CLI has been removed" in result.stderr
    assert "agent.guest_components_rest_api=all" in result.stderr
