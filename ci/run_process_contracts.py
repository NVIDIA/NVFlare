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

"""Run extracted process/transport scenarios serially in a fresh pytest interpreter.

The matrix and coverage jobs share this list so test moves cannot silently remove
coverage. Additional arguments are forwarded to pytest (for example --cov-append).
"""

import subprocess
import sys
from pathlib import Path

TESTS = (
    "tests/integration_test/fast/autofl_process_lifecycle_test.py",
    "tests/integration_test/fast/download_parallel_test.py",
    "tests/integration_test/fast/hf_distributed_contract_test.py",
    "tests/integration_test/fast/server_transit_e2e_test.py",
    "tests/integration_test/fast/streaming_pass_through_test.py",
)


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "pytest", "-q", "-rs", *sys.argv[1:], *TESTS],
            cwd=Path(__file__).resolve().parents[1],
        )
    )
