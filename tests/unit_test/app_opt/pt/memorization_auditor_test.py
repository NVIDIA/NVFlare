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

import random
import time

import numpy as np
import pytest
import torch

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.memorization_auditor import UnsupportedAudit
from nvflare.app_opt.pt.memorization_auditor import PTMemorizationAuditor, membership_auc


class TinyAuditor(PTMemorizationAuditor):
    def build_model(self):
        random.random()
        np.random.rand()
        return torch.nn.Linear(1, 1, bias=False)

    def evaluate(self, module, deadline):
        device = next(module.parameters()).device
        torch.rand(1, device=device)
        value = module(torch.ones(1, 1, device=device)).item()
        return {"scores": {"likelihood": {"members": [value, value + 1], "nonmembers": [0, 1]}}}


def test_diff_reconstruction_and_rng_isolation():
    with pytest.warns(Warning) as caught:
        auditor = TinyAuditor()
    for name in ("PTMemorizationAuditor", "MemorizationAuditor"):
        assert any(f"{name} API may change" in str(w.message) for w in caught)
    baseline = FLModel(params={"weight": torch.tensor([[2.0]])}, current_round=0)
    delta = FLModel(params={"weight": torch.tensor([[1.0]])}, params_type=ParamsType.DIFF, current_round=0)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    report = auditor.audit(delta, baseline, FLContext())
    assert report["metrics"]["likelihood/auc"] == 1.0
    assert report["metrics"]["likelihood/change_auc"] == 1.0
    assert report["sample_counts"]["likelihood"] == {"members": 2, "nonmembers": 2}
    assert torch.equal(delta.params["weight"], torch.tensor([[1.0]]))
    assert torch.equal(baseline.params["weight"], torch.tensor([[2.0]]))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_cuda_rng_isolation(device):
    before = torch.cuda.get_rng_state(device).clone()
    TinyAuditor(device=device).audit(FLModel(params={"weight": torch.tensor([[2.0]])}), None, FLContext())
    assert torch.equal(before, torch.cuda.get_rng_state(device))


def test_auc_orientation_and_ties():
    assert membership_auc([1, 2], [0, 1]) == 0.875
    assert membership_auc([0, 1], [1, 2]) == 0.125
    assert membership_auc([1], [1]) == 0.5
    with pytest.raises(UnsupportedAudit):
        membership_auc([], [1])
    with pytest.raises(ValueError):
        membership_auc([float("nan")], [1])


def test_timeout_and_partial_updates():
    auditor = TinyAuditor(max_seconds=0.001)
    auditor.evaluate = lambda *args: (time.sleep(0.01) or {"scores": {"x": {"members": [1], "nonmembers": [0]}}})
    model = FLModel(params={"weight": torch.tensor([[1.0]])})
    with pytest.raises(TimeoutError):
        auditor.audit(model, None, FLContext())
    with pytest.raises(UnsupportedAudit):
        TinyAuditor().audit(FLModel(params={"other": torch.tensor([1.0])}), None, FLContext())
