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

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "ci" / "should_run_full_ci.sh"


@pytest.mark.parametrize(
    ("changed_paths", "expected"),
    [
        (
            [
                "README.md",
                "docs/quickstart.rst",
                "docs/resources/federated_learning_overview.png",
                "examples/tutorials/README.md",
            ],
            "false",
        ),
        (["docs/llms.txt.in", "docs/_static/custom.css"], "false"),
        (["CONTRIBUTING.md", "CITATION.cff", "LICENSE"], "false"),
        (["docs/conf.py"], "true"),
        (["nvflare/apis/fl_context.py"], "true"),
        (["pyproject.toml"], "true"),
        ([".github/workflows/premerge.yml"], "true"),
        (["ci/should_run_full_ci.sh"], "true"),
        (["skills/nvflare-fed-stats/SKILL.md"], "true"),
        (["docs/quickstart.rst", "tests/unit_test/apis/fl_context_test.py"], "true"),
        ([], "true"),
    ],
)
def test_full_ci_selector(changed_paths, expected):
    completed = subprocess.run(
        ["bash", str(SELECTOR)],
        cwd=REPO_ROOT,
        input="\n".join(changed_paths),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
