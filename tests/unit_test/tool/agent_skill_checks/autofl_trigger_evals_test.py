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

from checks.lints import _eval_mentions_file_editing, _is_positive_eval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
AUTOFL_EVALS_PATH = REPO_ROOT / "dev_tools" / "agent" / "skill_evals" / "nvflare-autofl" / "evals.json"
NATURAL_PHRASING_EVAL_ID = "autofl-natural-phrasing-low-accuracy"


def _load_autofl_evals() -> list[dict]:
    data = json.loads(AUTOFL_EVALS_PATH.read_text(encoding="utf-8"))
    assert data["skill_name"] == "nvflare-autofl"
    return data["evals"]


def test_autofl_evals_include_natural_phrasing_positive_trigger():
    """A positive trigger eval must exist whose prompt never names Auto-FL explicitly.

    Guards the regression where the skill only triggered on explicit 'Use NVFLARE Auto-FL'
    phrasing and was missed for natural requests like 'accuracy is low, try two approaches'.
    """
    evals = _load_autofl_evals()
    case = next(item for item in evals if item["id"] == NATURAL_PHRASING_EVAL_ID)

    assert _is_positive_eval(case, "nvflare-autofl")
    assert case["nvflare"]["expected_skill"] == "nvflare-autofl"

    prompt = case["prompt"].lower()
    for expert_term in ("auto-fl", "autofl", "campaign", "candidate"):
        assert expert_term not in prompt
    # The prompt exercises the natural trigger phrases the skill description targets.
    assert "accuracy" in prompt
    assert "two approaches" in prompt
    assert "data and evaluation setup" in prompt


def test_autofl_natural_phrasing_eval_keeps_runner_and_cap_contract():
    evals = _load_autofl_evals()
    case = next(item for item in evals if item["id"] == NATURAL_PHRASING_EVAL_ID)

    mandatory_ids = {item["id"] for item in case["nvflare"]["mandatory_behavior"]}
    prohibited_ids = {item["id"] for item in case["nvflare"]["prohibited_behavior"]}

    assert "natural-phrasing-trigger" in mandatory_ids
    assert "campaign-runner-only" in mandatory_ids
    assert "candidate-cap-respected" in mandatory_ids
    assert "no-direct-project-optimization" in prohibited_ids

    # Trigger evals ship no fixtures, so the eval text must not describe file editing.
    assert case["files"] == []
    assert not _eval_mentions_file_editing(case)


def test_autofl_evals_keep_explicit_positive_and_negative_coverage():
    evals = _load_autofl_evals()
    ids = [item["id"] for item in evals]

    assert "autofl-optimize-existing-job" in ids
    assert "autofl-negative-pytorch-conversion" in ids
    assert "autofl-negative-diagnose-job" in ids
    assert "autofl-global-negative-web-app" in ids
    positives = [item for item in evals if _is_positive_eval(item, "nvflare-autofl")]
    assert len(positives) >= 2
