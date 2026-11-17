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
from nvflare.app_opt.pt.trap_loss import PTTRAPLoss


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


def test_recreated_reference_invalidates_cache_after_nonpersistent_buffer_change(tmp_path):
    class ScaledLM(TinyLM):
        def __init__(self):
            super().__init__()
            self.embed.register_buffer("scale", torch.ones(()), persistent=False)
            with torch.no_grad():
                self.embed.weight.copy_(torch.arange(8.0).expand(8, 8))

        def forward(self, input_ids, **kwargs):
            output = super().forward(input_ids, **kwargs)
            return SimpleNamespace(logits=output.logits * self.embed.scale)

    model = ScaledLM()
    assert "embed.scale" not in model.state_dict()
    inputs = torch.tensor([[1, 2, 3]])
    original = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    before, _ = original.score(inputs, inputs)
    unchanged = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    unchanged.score(inputs, inputs)
    assert unchanged.cache_hits == 1 and model.calls == 1

    model.embed.scale.mul_(2)
    with pytest.raises(ValueError, match="Reference changed"):
        original.score(inputs, inputs)
    recreated = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    actual, mask = recreated.score(inputs, inputs)
    logits = model.embed(inputs) * model.embed.scale
    expected = logits[:, :-1].log_softmax(-1).gather(-1, inputs[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, before)
    assert mask.all()
    assert recreated.reference_fingerprint != original.reference_fingerprint
    assert recreated.cache_hits == 0 and recreated.cache_misses == 1
    again, _ = recreated.score(inputs, inputs)
    torch.testing.assert_close(again, expected)
    assert recreated.cache_hits == 1 and model.calls == 2


def test_cached_optional_none_matches_online_and_has_distinct_key(tmp_path, monkeypatch):
    model = TinyLM()
    inputs = torch.tensor([[1, 2, 3]])
    forwarded = []
    forward = model.forward

    def record_inputs(input_ids, **kwargs):
        forwarded.append(kwargs)
        return forward(input_ids, **kwargs)

    monkeypatch.setattr(model, "forward", record_inputs)
    online = FrozenReferenceScorer(model, "tokenizer", "context")
    expected, expected_mask = online.score(inputs, inputs, attention_mask=None)
    cached = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    for _ in range(2):
        scores, mask = cached.score(inputs, inputs, attention_mask=None)
        torch.testing.assert_close(scores, expected)
        assert torch.equal(mask, expected_mask)
    assert forwarded == [{"attention_mask": None}, {"attention_mask": None}]
    assert cached.cache_hits == 1 and cached.cache_misses == 1

    # Explicit None, omitted inputs and tensor-valued masks must not share entries.
    cached.score(inputs, inputs)
    cached.score(inputs, inputs, attention_mask=torch.ones_like(inputs))
    assert model.calls == 4
    assert cached.cache_hits == 1 and cached.cache_misses == 3


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


@pytest.mark.parametrize("change", ["dtype", "storage"])
def test_reference_conversion_or_storage_replacement_rejects_cached_scores(tmp_path, change):
    model = TinyLM()
    scorer = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path)
    inputs = torch.tensor([[1, 2, 3]])
    scorer.score(inputs, inputs)
    if change == "dtype":
        model.half()
    else:
        model.embed.weight.data = torch.zeros_like(model.embed.weight)

    with pytest.raises(ValueError, match="Reference changed"):
        scorer.score(inputs, inputs)
    assert scorer.cache_hits == 0 and model.calls == 1


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("mutate", ["parameter", "buffer"])
def test_inference_reference_initialization_scoring_and_mutation_detection(tmp_path, use_cache, mutate):
    with torch.inference_mode():
        model = TinyLM()
        model.register_buffer("scale", torch.ones(1))
    inputs = torch.tensor([[1, 2, 3]])
    scorer = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path if use_cache else None)
    assert scorer.model is model
    assert torch.is_inference(model.embed.weight) and torch.is_inference(model.scale)
    expected = model.embed(inputs)[:, :-1].log_softmax(-1).gather(-1, inputs[:, 1:, None]).squeeze(-1)
    for _ in range(2):
        scores, mask = scorer.score(inputs, inputs)
        torch.testing.assert_close(scores, expected)
        assert mask.all() and not scores.requires_grad
    assert model.calls == (1 if use_cache else 2)
    assert scorer.cache_hits == int(use_cache)

    # Inference tensors can still be changed inside inference_mode, without counters.
    with torch.inference_mode():
        tensor = model.embed.weight if mutate == "parameter" else model.scale
        tensor.add_(1)
    with pytest.raises(ValueError, match="Reference changed"):
        scorer.score(inputs, inputs)
    assert model.calls == (1 if use_cache else 2)
    assert scorer.cache_hits == int(use_cache)


def test_online_scoring_does_not_hash_inputs(monkeypatch):
    model = TinyLM()
    scorer = FrozenReferenceScorer(model, "tokenizer", "context")
    monkeypatch.setattr(reference_scorer, "tensor_digest", lambda *_: pytest.fail("online inputs were hashed"))
    inputs = torch.tensor([[1, 2, 3]])
    scores, mask = scorer.score(inputs, inputs)
    assert torch.isfinite(scores).all() and mask.all()
    assert model.calls == 1


@pytest.mark.parametrize("use_cache", [False, True])
def test_inference_scoring_mask_can_be_used_for_training(tmp_path, use_cache):
    model = TinyLM()
    scorer = FrozenReferenceScorer(model, "tokenizer", "context", tmp_path if use_cache else None)
    inputs = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, -100, 4]])
    loss = PTTRAPLoss(weight=0.5, threshold=0.5)
    for _ in range(2):
        with torch.inference_mode():
            reference, mask = scorer.score(inputs, labels)
        assert torch.is_inference(mask)
        target = (reference.clone() + torch.tensor([[1.0, 99.0, -1.0]])).requires_grad_(True)
        penalty = loss(target, reference, mask)
        penalty.backward()
        assert penalty.item() == pytest.approx(0.125)
        torch.testing.assert_close(target.grad, torch.tensor([[0.25, 0.0, 0.0]]))
        assert torch.equal(mask, torch.tensor([[True, False, True]]))
    assert model.calls == (1 if use_cache else 2)
    assert scorer.cache_hits == int(use_cache)
