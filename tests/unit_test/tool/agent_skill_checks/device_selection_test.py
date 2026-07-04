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
        ("nvflare-convert-pytorch", "pytorch-device-selection"),
        ("nvflare-convert-lightning", "lightning-device-selection"),
    ]:
        evals_path = EVAL_ROOT / skill_name / "evals.json"
        data = json.loads(evals_path.read_text(encoding="utf-8"))
        case = next(item for item in data["evals"] if item["id"] == case_id)
        behavior_ids = {item["id"] for item in case["nvflare"]["mandatory_behavior"] if isinstance(item, dict)}

        assert data["skill_name"] == skill_name
        assert "-convert-" in data["skill_name"]
        assert DEVICE_SELECTION_BEHAVIOR_ID in behavior_ids
        assert "preserve-device-intent" not in behavior_ids


def test_basic_cpu_fixtures_do_not_claim_gpu_conditional_behavior():
    for skill_name, case_id in [
        ("nvflare-convert-pytorch", "pytorch-convert-basic"),
        ("nvflare-convert-lightning", "lightning-convert-basic"),
    ]:
        data = json.loads(EVAL_ROOT.joinpath(skill_name, "evals.json").read_text(encoding="utf-8"))
        case = next(item for item in data["evals"] if item["id"] == case_id)
        behavior_ids = {item["id"] for item in case["nvflare"]["mandatory_behavior"] if isinstance(item, dict)}

        assert DEVICE_SELECTION_BEHAVIOR_ID not in behavior_ids


@pytest.mark.parametrize(
    ("skill_name", "case_id"),
    [
        ("nvflare-convert-pytorch", "pytorch-device-selection"),
        ("nvflare-convert-lightning", "lightning-device-selection"),
    ],
)
def test_device_selection_eval_fixture_is_applicable(skill_name, case_id):
    data = json.loads(EVAL_ROOT.joinpath(skill_name, "evals.json").read_text(encoding="utf-8"))
    case = next(item for item in data["evals"] if item["id"] == case_id)
    source = "\n".join(
        EVAL_ROOT.joinpath(skill_name, rel_path).read_text(encoding="utf-8") for rel_path in case["files"]
    )

    assert source_uses_gpu_when_available(source) is True


@pytest.mark.parametrize(
    "source",
    [
        PYTORCH_SOURCE,
        LIGHTNING_SOURCE,
    ],
)
def test_source_uses_gpu_when_available_detects_pytorch_and_lightning(source):
    assert source_uses_gpu_when_available(source) is True


@pytest.mark.parametrize(
    ("gpu_available", "expected_evidence"),
    [
        (True, "GPU-available branch selects CUDA/GPU"),
        (False, "GPU-unavailable branch selects the CPU fallback"),
    ],
)
def test_device_selection_status_pass_for_both_availability_branches(gpu_available, expected_evidence):
    result = check_device_selection(PYTORCH_SOURCE, GENERATED_CONDITIONAL, gpu_available=gpu_available)

    assert result.status == "pass"
    record = result.as_behavior_record()
    assert record["mandatory_behavior"][DEVICE_SELECTION_BEHAVIOR_ID]["status"] == "pass"
    assert "torch.cuda.is_available" in record["mandatory_behavior"][DEVICE_SELECTION_BEHAVIOR_ID]["evidence"]
    assert expected_evidence in result.evidence


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
    assert "no AST-detectable device-selection logic" in result.evidence


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


def test_device_selection_does_not_trust_forged_runtime_gpu_log():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
def train(model, loader):
    return None
""",
        runtime_log="validation selected device=cuda:0",
        gpu_available=True,
    )

    assert result.status == "missing"
    assert "Raw runtime log text was not trusted" in result.evidence


def test_device_selection_does_not_let_forged_runtime_cpu_log_override_ast():
    result = check_device_selection(
        PYTORCH_SOURCE,
        GENERATED_CONDITIONAL,
        runtime_log="validation selected device=cpu",
        gpu_available=True,
    )

    assert result.status == "pass"
    assert "GPU-available branch selects CUDA/GPU" in result.evidence


def test_source_detection_ignores_regex_like_comments_and_strings():
    source = """
# torch.cuda.is_available() ? cuda : cpu
message = "torch.cuda.is_available() device=cuda fallback=cpu"
device = "cpu"
"""

    assert source_uses_gpu_when_available(source) is False


def test_source_detection_supports_if_statement_and_import_alias():
    source = """
import torch as pt

if pt.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
"""

    assert source_uses_gpu_when_available(source) is True


def test_device_selection_status_fail_when_generated_hard_codes_gpu():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
import torch

device = torch.device("cuda:0")
""",
    )

    assert result.status == "fail"
    assert "hard-codes GPU" in result.evidence


def test_device_selection_detects_hardcoded_device_through_torch_alias():
    result = check_device_selection(
        PYTORCH_SOURCE,
        """
import torch as pt

device = pt.device("cpu")
""",
    )

    assert result.status == "fail"
    assert "hard-codes CPU" in result.evidence


def test_device_selection_rejects_availability_call_in_unrelated_or_condition():
    generated = """
import torch

if torch.cuda.is_available() or True:
    device = "cuda"
else:
    device = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_rejects_reassigned_availability_alias():
    generated = """
import torch

has_cuda = torch.cuda.is_available()
has_cuda = True
if has_cuda:
    device = "cuda"
else:
    device = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "missing"
    assert "reachable, same-target" in result.evidence


def test_device_selection_rejects_cross_scope_availability_alias():
    generated = """
import torch

def inspect_environment():
    has_cuda = torch.cuda.is_available()

def train():
    if has_cuda:
        device = "cuda"
    else:
        device = "cpu"
"""

    assert source_uses_gpu_when_available(generated) is False


def test_device_selection_rejects_unreachable_nested_gpu_assignment():
    generated = """
import torch

if torch.cuda.is_available():
    if False:
        device = "cuda"
else:
    device = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_rejects_branch_selection_overwritten_by_unknown_value():
    generated = """
import torch

if torch.cuda.is_available():
    device = "cuda"
    device = choose_device()
else:
    device = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_rejects_conditional_selection_overwritten_by_cpu():
    generated = """
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "fail"


def test_device_selection_rejects_unused_conditional_decoy_with_real_cpu_selection():
    generated = """
import torch

def unused_decoy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "fail"


def test_device_selection_allows_separate_cpu_only_validation_scope():
    generated = """
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def validate_on_cpu():
    device = torch.device("cpu")
    return device
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "pass"


def test_device_selection_rejects_assignment_after_return():
    generated = """
import torch

def choose_device():
    if torch.cuda.is_available():
        return None
        device = "cuda"
    else:
        device = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_requires_gpu_and_cpu_to_update_same_target():
    generated = """
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    accelerator = "cpu"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "missing"


def test_device_selection_accepts_reachable_default_then_gpu_override():
    generated = """
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status == "pass"


def test_device_selection_rejects_rebound_torch_alias():
    generated = """
import torch

torch = object()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
"""

    assert source_uses_gpu_when_available(generated) is False


def test_device_selection_rejects_mutated_torch_cuda_attribute():
    generated = """
import torch

torch.cuda.is_available = lambda: True
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
"""

    assert source_uses_gpu_when_available(generated) is False


def test_device_selection_rejects_stale_default_after_augmented_assignment():
    generated = """
import torch

device = "cpu"
device += get_device_suffix()
if torch.cuda.is_available():
    device = "cuda"
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_rejects_stale_to_default_after_receiver_rebind():
    generated = """
import torch

model.to("cpu")
model = replacement_model()
if torch.cuda.is_available():
    model.to("cuda")
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"


def test_device_selection_does_not_treat_none_as_verified_cpu_fallback():
    generated = """
import torch

accelerator = "gpu" if torch.cuda.is_available() else None
"""

    result = check_device_selection(PYTORCH_SOURCE, generated)

    assert result.status != "pass"
