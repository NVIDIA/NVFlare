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

from dev_tools.agent.skills.checks.lints import LINT_SKILL_POLICY_COVERAGE, run_v1_lints

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTORCH_POLICY_OWNERS = {
    "nvflare-convert-lightning",
    "nvflare-convert-pytorch",
    "nvflare-shared",
}

NVFLARE_POLICY_TEST_EVIDENCE = [
    {
        "id": "no-cross-family-pytorch-model-exchange-reference",
        "description": "non-PyTorch framework skills do not reference or load the shared PyTorch model-exchange guidance",
        "polarity": "prohibited",
        "test": "test_non_pytorch_skills_do_not_reference_pytorch_model_exchange",
    }
]


def test_non_pytorch_skills_do_not_reference_pytorch_model_exchange():
    scanned = []
    for skill_dir in sorted(REPO_ROOT.joinpath("skills").iterdir()):
        if not skill_dir.is_dir() or skill_dir.name in PYTORCH_POLICY_OWNERS:
            continue
        for path in sorted(skill_dir.rglob("*")):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            scanned.append(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "pytorch-model-exchange.md" not in text, f"cross-family PyTorch guidance reference in {path}"

    assert scanned, "expected at least one non-PyTorch skill guidance file"


def test_shipped_skills_have_verified_policy_coverage():
    result = run_v1_lints(
        REPO_ROOT / "skills",
        evals_root=REPO_ROOT / "dev_tools" / "agent" / "skill_evals",
        checks=[LINT_SKILL_POLICY_COVERAGE],
    )

    assert result["status"] == "ok", result["findings"]
    assert result["findings"] == []
