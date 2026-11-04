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

import os
import sys
from pathlib import Path

import pytest

from tests.process_test_utils import run_owned_process


@pytest.mark.timeout(90)
def test_reliable_tcp_retry_and_mpm_shutdown():
    root = str(Path(__file__).resolve().parents[3])
    result = run_owned_process(
        [
            sys.executable,
            "-c",
            "from tests.f3_cleanup_test_utils import reliable_cleanup_flow; reliable_cleanup_flow()",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"retry_ack": true' in result.stdout
    assert "MPM cleanup released all receive tasks and scheduler threads" in result.stdout
