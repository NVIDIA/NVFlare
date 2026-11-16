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
import subprocess
import sys
from pathlib import Path

import pytest

from tests.process_test_utils import run_owned_process
from tests.timing_utils import expire_deadline_when_ready


@pytest.mark.timeout(60)
@pytest.mark.parametrize("scenario", ["concurrent_run_results", "concurrent_downloadable_release"])
def test_concurrent_cleanup_has_process_watchdog(scenario):
    root = str(Path(__file__).resolve().parents[3])
    result = run_owned_process(
        [sys.executable, "-c", f"from tests.concurrent_test_utils import {scenario}; {scenario}()"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.timeout(40)
@pytest.mark.parametrize(
    "scenario, module, name, method",
    [
        ("concurrent_run_results", "nvflare.recipe.run", "Run", "get_result"),
        ("concurrent_downloadable_release", "nvflare.app_common.np.np_downloader", "ArrayDownloadable", "release"),
    ],
)
def test_product_deadlock_is_stopped_by_parent_watchdog(tmp_path, monkeypatch, scenario, module, name, method):
    from tests import process_test_utils

    root = str(Path(__file__).resolve().parents[3])
    ready = tmp_path / "deadlock-entered"
    expire_deadline_when_ready(monkeypatch, process_test_utils, ready.exists)
    code = (
        "import threading; from pathlib import Path; "
        f"from {module} import {name}; "
        "gate = threading.Event(); "
        f"{name}.{method} = lambda *a, **k: "
        f"(print('deadlock entered', flush=True), Path({str(ready)!r}).touch(), gate.wait()); "
        f"from tests.concurrent_test_utils import {scenario}; {scenario}()"
    )
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_owned_process([sys.executable, "-c", code], cwd=root, env={**os.environ, "PYTHONPATH": root}, timeout=5)
    assert "deadlock entered" in caught.value.output
