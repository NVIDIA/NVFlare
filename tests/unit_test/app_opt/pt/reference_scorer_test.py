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

import weakref
from types import SimpleNamespace

import pytest
import torch

from nvflare.app_opt.pt import reference_scorer
from nvflare.app_opt.pt.reference_scorer import FrozenReferenceScorer


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 8)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=self.embed(input_ids))


def test_cache_equivalence_and_input_tokenizer_context_invalidation(tmp_path):
    model = TinyLM()
    inputs = torch.tensor([[1, 2, 3]])
    with pytest.warns(Warning, match="FrozenReferenceScorer API may change"):
        scorer = FrozenReferenceScorer(model, "tokenizer1", "context1", tmp_path)
    first, mask = scorer.score(inputs, inputs)
    again, again_mask = scorer.score(inputs, inputs)
    torch.testing.assert_close(first, again)
    assert torch.equal(mask, again_mask)
    assert model.calls == 1
    assert all(not p.requires_grad for p in model.parameters())
    assert not first.requires_grad
    scorer.score(inputs, torch.tensor([[-100, -100, 3]]))
    FrozenReferenceScorer(model, "tokenizer2", "context1", tmp_path).score(inputs, inputs)
    FrozenReferenceScorer(model, "tokenizer1", "context2", tmp_path).score(inputs, inputs)
    assert model.calls == 4
    with torch.no_grad():
        model.embed.weight.add_(1)
    with pytest.raises(ValueError, match="Reference changed"):
        scorer.score(inputs, inputs)
    FrozenReferenceScorer(model, "tokenizer1", "context1", tmp_path).score(inputs, inputs)
    assert model.calls == 5


def test_replaced_tensor_cannot_reuse_cached_scores_when_object_id_is_recycled(tmp_path, monkeypatch):
    # Force an ID collision to reproduce allocator reuse deterministically.
    monkeypatch.setattr(reference_scorer, "id", lambda tensor: 0, raising=False)
    model = TinyLM()
    model.embed.weight = torch.nn.Parameter(torch.zeros(8, 8))
    scorer = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    inputs = torch.tensor([[1, 2, 3]])
    scorer.score(inputs, inputs)
    previous = weakref.ref(model.embed.weight)
    model.embed.weight = torch.nn.Parameter(torch.ones(8, 8), requires_grad=False)
    assert previous() is None
    with pytest.raises(ValueError, match="Reference changed"):
        scorer.score(inputs, inputs)
    assert scorer.cache_hits == 0
