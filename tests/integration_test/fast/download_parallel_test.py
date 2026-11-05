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

"""Real parallel downloads with a bounded, reaped worker process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(90)
def test_parallel_downloads_complete_with_real_streaming():
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "tests.parallel_download_test_utils"],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            timeout=75,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"parallel download worker timed out: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
