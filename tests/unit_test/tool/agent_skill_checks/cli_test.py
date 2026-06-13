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

import json
import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks import cli  # noqa: E402


def test_agent_skill_checks_cli_emits_json_and_nonzero_on_findings(tmp_path, capsys):
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    exit_code = cli.main(["--skills-root", str(skills_root / "missing"), "--format", "json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["summary"]["error_count"] == 1
    assert payload["findings"][0]["id"] == "skill-frontmatter-lint"


def test_agent_skill_checks_cli_emits_text(capsys, tmp_path):
    exit_code = cli.main(["--skills-root", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "agent skill checks:" in capsys.readouterr().out


def test_agent_skill_checks_cli_reports_invalid_check_as_json_error(capsys, tmp_path):
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    exit_code = cli.main(["--skills-root", str(skills_root), "--format", "json", "--check", "bad-id"])

    assert exit_code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1"
    assert payload["status"] == "error"
    assert payload["passed"] is False
    assert payload["error_code"] == "INVALID_ARGS"
    assert "bad-id" in payload["message"]
