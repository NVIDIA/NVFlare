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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Check both MIL variants with synthetic feature bags, without clinical inputs."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import aggregation, client_training, training
from core.mil_models import build_mil_model
from core.topk_mil import DTFDTopKMIL, forward_topk_bags
from fabric_common import load_setup


def test_topk_uses_class_scores_and_passes_selected_patches_to_tier_two():
    model = DTFDTopKMIL(2, embed_dim=2, attn_dim=2, pseudo_bags=2, dropout=0, top_k=2).eval()
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.eye(2))
        model.encoder[0].bias.zero_()
        for parameter in model.local_attention.parameters():
            parameter.zero_()
        model.pseudo_classifier.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
        model.pseudo_classifier.bias.zero_()
    data = torch.stack((torch.arange(1, 13, dtype=torch.float32), torch.ones(12)), dim=1).unsqueeze(0)
    captured = []
    hook = model.global_attention.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        for descending in (True, False):
            details = model(data, return_attention=True)["topk_details"][0]
            for group, selected in zip(details["pseudo_bag_indices"], details["selected_patch_indices"], strict=True):
                ordered = group.sort(descending=descending).values
                assert torch.equal(selected, torch.cat((ordered[:2], ordered[-2:])))
            selected = torch.cat(details["selected_patch_indices"])
            assert captured[-1].shape == (8, 2)
            assert torch.equal(captured[-1], data[0, selected])
            # Uniform attention cannot explain a ranking reversal caused only by the classifier.
            with torch.no_grad():
                model.pseudo_classifier.weight.mul_(-1)
    finally:
        hook.remove()


@pytest.mark.parametrize("training_mode", [False, True])
@pytest.mark.parametrize("top_k", [1, 2, 100])
def test_small_bags_use_both_tiers_without_duplicate_patches(training_mode, top_k):
    model = DTFDTopKMIL(4, embed_dim=8, attn_dim=4, dropout=0, top_k=top_k).train(training_mode)
    for count in (1, 2, 7, 8, 9, 17):
        calls = []
        hook = model.global_attention.register_forward_pre_hook(
            lambda module, args, seen=calls: seen.append(len(args[0]))
        )
        try:
            result = model(torch.rand(1, count, 4), return_attention=True)
        finally:
            hook.remove()
        detail = result["topk_details"][0]
        groups = detail["pseudo_bag_indices"]
        assert 1 <= len(groups) <= min(8, count)
        assert torch.equal(torch.cat(groups).sort().values, torch.arange(count))
        selected = torch.cat(detail["selected_patch_indices"])
        assert len(selected) == selected.unique().numel() == sum(min(2 * top_k, len(group)) for group in groups)
        assert calls == [len(selected)]
        assert result["logits"].shape == (1, 2)
        assert torch.isfinite(result["logits"]).all()
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty(1, 0, 4))


def test_topk_prediction_is_repeatable_and_survives_checkpoint_roundtrip(tmp_path):
    model = DTFDTopKMIL(4, embed_dim=8, attn_dim=4, top_k=2).eval()
    data = torch.randn(2, 31, 4)
    rng = torch.get_rng_state().clone()
    first = model(data)["logits"]
    assert torch.equal(rng, torch.get_rng_state())
    torch.rand(11)
    assert torch.equal(first, model(data)["logits"])
    assert torch.equal(first, model(data.flip(0))["logits"].flip(0))
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = DTFDTopKMIL(4, embed_dim=8, attn_dim=4, top_k=2).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    assert torch.equal(first, restored(data)["logits"])


def test_auxiliary_loss_weights_patients_not_pseudo_bag_counts():
    pseudo_logits = [
        torch.tensor([[1.0, 2.0]], requires_grad=True),
        torch.tensor([[2.0, 1.0], [0.0, 1.0], [3.0, 2.0]], requires_grad=True),
    ]
    outputs = iter(pseudo_logits)

    def model(data):
        logits = next(outputs)
        return {"logits": logits.mean(0, keepdim=True), "pseudo_logits": [logits]}

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))
    _, auxiliary = forward_topk_bags(
        model, [torch.zeros(1, 2), torch.zeros(3, 2)], torch.tensor([0, 1]), criterion, torch.device("cpu")
    )
    expected = (
        F.cross_entropy(pseudo_logits[0], torch.tensor([0]))
        + 3 * F.cross_entropy(pseudo_logits[1], torch.ones(3, dtype=torch.long))
    ) / 4
    torch.testing.assert_close(auxiliary, expected)
    auxiliary.backward()
    assert all(logits.grad is not None and torch.isfinite(logits.grad).all() for logits in pseudo_logits)


def test_gradients_reach_projection_and_both_mil_tiers():
    model = build_mil_model("dtfd_topk", 8, embed_dim=16, attn_dim=8, dropout=0)
    labels = torch.tensor([0, 1, 0])
    bags = [torch.rand(count, 8) for count in (37, 48, 29)]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))
    logits, auxiliary = forward_topk_bags(model, bags, labels, criterion, torch.device("cpu"))
    (criterion(logits, labels) + auxiliary).backward()
    for prefix in ("encoder", "local_attention", "global_attention", "classifier", "pseudo_classifier"):
        gradients = [parameter.grad for name, parameter in model.model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum().item() for value in gradients) > 0, prefix


@pytest.mark.parametrize("model_name", ["dtfd_mil", "dtfd_topk"])
def test_local_training_and_prediction_on_synthetic_bags(model_name, monkeypatch):
    args = SimpleNamespace(
        embed_dim=16,
        attn_dim=8,
        dtfd_pseudo_bags=8,
        dropout=0.25,
        dtfd_top_k=2,
        dtfd_eval_group_seed=42,
        dtfd_pseudo_loss_weight=1.0,
        lr=1e-3,
        weight_decay=1e-4,
        local_epochs=2,
        amp=False,
        threshold=0.5,
        model_variant="topk" if model_name == "dtfd_topk" else "pooling",
    )
    bags = [torch.randn(count, 8) for count in (7, 19, 31)]
    labels = torch.tensor([0, 1, 0])
    ids = [f"synthetic-{index}" for index in range(3)]
    loader = [(bags, labels, ids)]
    frame = pd.DataFrame({"patient_id": ids, "recurrence_label": labels.tolist()})
    for column in training.PREDICTION_METADATA:
        if column not in frame:
            frame[column] = 1 if column == "num_slides" else "synthetic"
    monkeypatch.setattr(client_training, "make_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(training, "make_loader", lambda *args, **kwargs: loader)
    model = training.build_model(model_name, 8, args)
    initial = aggregation.copy_state_dict(model.state_dict())
    state, log = client_training.train_local_client_final(
        SimpleNamespace(algorithm="fedavg"), initial, frame, model_name, 8, args, torch.device("cpu"), seed=123
    )
    assert not torch.equal(initial["model.classifier.1.weight"], state["model.classifier.1.weight"])
    if model_name == "dtfd_topk":
        assert log["loss"] == pytest.approx(log["patient_loss"] + log["pseudo_loss"], abs=1e-6)
        assert not torch.equal(initial["model.pseudo_classifier.weight"], state["model.pseudo_classifier.weight"])
    model.load_state_dict(state, strict=True)
    calls = []
    hook = model.model.global_attention.register_forward_pre_hook(lambda module, args: calls.append(1))
    try:
        _, predictions = training.evaluate_model(model, frame, frame, args, seed=1, device=torch.device("cpu"))
    finally:
        hook.remove()
    assert predictions.patient_id.tolist() == ids
    if model_name == "dtfd_topk":
        assert len(calls) == len(bags)


@pytest.mark.parametrize("encoder", ["uni", "virchow2"])
@pytest.mark.parametrize("variant", ["pooling", "topk"])
def test_encoder_dimensions_and_variant_settings(make_fold, encoder, variant):
    case = make_fold(encoder=encoder, variant=variant)
    pooling = load_setup(case.root, encoder)
    for key in ("settings", "folds", "sites", "encoder", "input_dim", "nvflare_version"):
        assert case.setup[key] == pooling[key]
    model = build_mil_model(case.setup["model_name"], case.setup["input_dim"]).eval()
    with torch.no_grad():
        result = model(torch.randn(1, 24, case.setup["input_dim"]))
    assert result["logits"].shape == (1, 2)
    assert model.model.encoder[0].out_features == 512
    if variant == "topk":
        assert result["pseudo_logits"][0].shape == (8, 2)
