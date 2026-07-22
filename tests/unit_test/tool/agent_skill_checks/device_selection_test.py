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
from pathlib import Path

import pytest

from dev_tools.agent.skills.checks.device_selection import (
    DEVICE_SELECTION_BEHAVIOR_ID,
    check_device_selection,
    source_uses_gpu_when_available,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = REPO_ROOT / "dev_tools" / "agent" / "skill_evals"

PYTORCH_SOURCE = """
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""

LIGHTNING_SOURCE = """
import torch
from pytorch_lightning import Trainer

trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
"""

GENERATED_CONDITIONAL = """
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""


def test_convert_eval_declarations_use_device_selection_behavior_id():
    for skill_name, case_id in [
        ("nvflare-convert-pytorch", "pytorch-convert-basic"),
        ("nvflare-convert-lightning", "lightning-convert-basic"),
    ]:
        evals_path = EVAL_ROOT / skill_name / "evals.json"
        data = json.loads(evals_path.read_text(encoding="utf-8"))
        case = next(item for item in data["evals"] if item["id"] == case_id)
        behavior_ids = {item["id"] for item in case["nvflare"]["mandatory_behavior"] if isinstance(item, dict)}

        assert data["skill_name"] == skill_name
        assert "-convert-" in data["skill_name"]
        assert DEVICE_SELECTION_BEHAVIOR_ID in behavior_ids
        assert "preserve-device-intent" not in behavior_ids


@pytest.mark.parametrize(
    "source",
    [
        PYTORCH_SOURCE,
        LIGHTNING_SOURCE,
    ],
)
def test_source_uses_gpu_when_available_detects_pytorch_and_lightning(source):
    assert source_uses_gpu_when_available(source) is True


def test_device_selection_status_pass_when_generated_keeps_cuda_availability():
    result = check_device_selection(PYTORCH_SOURCE, GENERATED_CONDITIONAL)

    assert result.status == "pass"
    record = result.as_behavior_record()
    assert record["mandatory_behavior"][DEVICE_SELECTION_BEHAVIOR_ID]["status"] == "pass"
    assert "torch.cuda.is_available" in record["mandatory_behavior"][DEVICE_SELECTION_BEHAVIOR_ID]["evidence"]


def test_device_selection_status_fail_when_generated_hard_codes_cpu():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
import torch

device = torch.device("cpu")
""",
    )

    assert result.status == "fail"
    assert "hard-codes CPU" in result.evidence


def test_device_selection_status_missing_when_generated_has_no_device_logic():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
def train(model, loader):
    for batch in loader:
        model(batch)
""",
    )

    assert result.status == "missing"
    assert "no detectable device-selection logic" in result.evidence


def test_device_selection_status_not_applicable_for_cpu_only_source():
    result = check_device_selection(
        """
import torch

device = torch.device("cpu")
""",
        """
import torch

device = torch.device("cpu")
""",
    )

    assert result.status == "not_applicable"
    assert "source does not select CUDA/GPU conditionally" in result.evidence


def test_device_selection_runtime_gpu_evidence_can_pass_static_missing_code():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
def train(model, loader):
    return None
""",
        runtime_log="validation selected device=cuda:0",
        gpu_available=True,
    )

    assert result.status == "pass"
    assert "runtime log selected" in result.evidence


def test_device_selection_runtime_gpu_evidence_fails_cpu_selection():
    result = check_device_selection(
        PYTORCH_SOURCE,
        GENERATED_CONDITIONAL,
        runtime_log="validation selected device=cpu",
        gpu_available=True,
    )

    assert result.status == "fail"
    assert "while GPU was available" in result.evidence
