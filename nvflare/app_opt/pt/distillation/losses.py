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

"""Temperature-scaled classification distillation losses."""

import math

import torch
from torch.nn import functional as F


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_output: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    teacher_is_logits: bool = True,
) -> torch.Tensor:
    """Return T**2 * KL(teacher || student), without teacher gradients.

    Inputs are floating tensors with matching [batch, classes] shapes and class
    order, on the same device. Student inputs are always raw logits. By default
    teacher inputs are raw logits too. With teacher_is_logits=False, supply
    probabilities already softened at T; this function does not soften them
    again. Their rows must be nonnegative and sum to one.

    batchmean sums over classes and divides by batch size; sum sums every
    entry; none returns [batch, classes] terms, which may individually be
    negative. Arithmetic uses native PyTorch dtypes, without precision conversion
    or overflow recovery. A scalar batchmean loss is the trainer's default.
    """
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if reduction not in ("none", "sum", "batchmean"):
        raise ValueError("reduction must be none, sum or batchmean")
    if not isinstance(teacher_is_logits, bool):
        raise TypeError("teacher_is_logits must be a boolean")
    for value in (student_logits, teacher_output):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not all(value.shape):
            raise ValueError("outputs must be nonempty [batch, classes] tensors")
        if not value.is_floating_point():
            raise TypeError("outputs must be floating tensors")
    if student_logits.shape != teacher_output.shape or student_logits.device != teacher_output.device:
        raise ValueError("student and teacher outputs must have matching shapes and devices")
    target = teacher_output.detach()
    if teacher_is_logits:
        target = F.softmax(target / temperature, dim=1)
    elif (
        not torch.isfinite(target).all()
        or (target < 0).any()
        or not torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]), rtol=1e-4, atol=1e-6)
    ):
        raise ValueError("teacher probabilities must be finite, nonnegative and sum to one per row")
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=1), target, reduction=reduction) * temperature**2
