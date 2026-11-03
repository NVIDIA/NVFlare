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

from pathlib import Path

import pytest

from nvflare.apis.workspace import Workspace


@pytest.mark.parametrize(
    "audit_root, log_root, result_root, expected_root",
    [
        (False, False, False, None),
        (False, False, True, "result"),
        (False, True, False, "log"),
        (False, True, True, "log"),
        (True, False, False, "audit"),
        (True, False, True, "audit"),
        (True, True, False, "audit"),
        (True, True, True, "audit"),
    ],
)
def test_audit_paths_follow_site_output_root_precedence(
    tmp_path, monkeypatch, audit_root, log_root, result_root, expected_root
):
    for name, enabled in (("audit", audit_root), ("log", log_root), ("result", result_root)):
        variable = f"NVFL_{name.upper()}_ROOT"
        if enabled:
            monkeypatch.setenv(variable, str(tmp_path / name))
        else:
            monkeypatch.delenv(variable, raising=False)

    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    workspace = Workspace(str(root), site_name="site-1")

    expected = tmp_path / expected_root / "site-1" if expected_root else root
    assert Path(workspace.get_audit_file_path()) == expected / "audit.log"
    assert Path(workspace.get_audit_file_path("job-1")) == expected / "job-1" / "audit.log"
    # Resolving a destination does not initialize an auditor or create an audit file.
    assert not (expected / "audit.log").exists()
    assert not (expected / "job-1" / "audit.log").exists()
