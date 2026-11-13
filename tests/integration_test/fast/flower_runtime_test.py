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

"""Real SuperLink, SuperNodes, app workers and ReliableMessage; selected by flower-runtime CI.

Requires the full flwr>=1.43 runtime on Python 3.11+. No GPU or data download.
"""

import sys
from pathlib import Path

from tests.process_test_utils import run_owned_process


def test_real_flower_round_through_nvflare(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    result = run_owned_process(
        [sys.executable, "-m", "tests.flower_runtime_test_utils", str(tmp_path), str(repo / "tests/data/flower_http")],
        cwd=str(repo),
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    logs = list(tmp_path.rglob("server_app_log.txt"))
    assert len(logs) == 1, result.stdout + result.stderr
    assert "FLOWER_RUNTIME_ROUND_OK clients=2" in logs[0].read_text(), logs[0].read_text()
