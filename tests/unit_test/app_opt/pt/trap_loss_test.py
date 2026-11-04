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

import pytest
import torch

from nvflare.app_opt.pt.trap_loss import PTTRAPLoss, causal_token_log_probs


def test_values_gradients_and_masked_nonfinite_padding():
    target = torch.tensor([-1.0, -3.0, float("nan")], requires_grad=True)
    reference = torch.tensor([-3.0, -2.0, float("nan")], requires_grad=True)
    penalty = PTTRAPLoss(0.5, 0.5)(target, reference, torch.tensor([True, True, False]))
    assert penalty.item() == pytest.approx(0.375)
    penalty.backward()
    torch.testing.assert_close(target.grad, torch.tensor([0.25, 0.0, 0.0]))
    assert reference.grad is None


@pytest.mark.parametrize("weight,mask", [(0.0, [True, True]), (1.0, [False, False])])
def test_disabled_or_empty_is_differentiable_zero(weight, mask):
    target = torch.tensor([-2.0, -1.0], requires_grad=True)
    value = PTTRAPLoss(weight)(target, target - 1, torch.tensor(mask))
    value.backward()
    assert value.item() == 0
    assert torch.equal(target.grad, torch.zeros_like(target))


def test_causal_alignment_and_ce_equivalence():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 4, 5, 6, 1]])
    probs, mask = causal_token_log_probs(logits, labels)
    actual = -probs[mask].mean()
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)


def test_accumulation_uses_token_counts_not_microbatch_means():
    target = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    ref, mask = target.detach() - 1, torch.ones(3, dtype=torch.bool)
    loss = PTTRAPLoss()
    combined = loss(target[:1], ref[:1], mask[:1], normalizer=3) + loss(target[1:], ref[1:], mask[1:], normalizer=3)
    torch.testing.assert_close(combined, loss(target, ref, mask))


def test_weight_above_one_reverses_active_ce_gradient():
    target = torch.tensor([-1.0], requires_grad=True)
    reference = torch.tensor([-2.0])
    objective = -target.mean() + PTTRAPLoss(2.0)(target, reference, torch.tensor([True]))
    objective.backward()
    assert target.grad.item() == 1.0


def test_shape_and_mask_errors():
    with pytest.raises(ValueError, match="shapes"):
        PTTRAPLoss()(torch.zeros(2), torch.zeros(3), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        PTTRAPLoss()(torch.zeros(2), torch.zeros(2), torch.ones(2))
    with pytest.raises(ValueError, match="normalizer"):
        PTTRAPLoss()(torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool), normalizer=1)


@pytest.mark.parametrize("weight,threshold", [(-1, 0), (1, -1), (float("nan"), 0), (1, float("inf"))])
def test_invalid_configuration(weight, threshold):
    with pytest.raises(ValueError):
        PTTRAPLoss(weight, threshold)
