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

import importlib.util
import os
import sys
from collections import OrderedDict

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _example_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "integration", "nemo", "examples", "peft")
    )


def _load_evaluate_module():
    example_dir = _example_dir()
    sys.path.insert(0, example_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "nemo_peft_evaluate_sentiment", os.path.join(example_dir, "evaluate_sentiment.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_dir)


def _load_assess_module():
    example_dir = _example_dir()
    spec = importlib.util.spec_from_file_location(
        "nemo_peft_assess_validation", os.path.join(example_dir, "assess_validation.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the evaluator")
def test_evaluate_sentiment_summarizes_scores_and_validation_bias():
    evaluate_sentiment = _load_evaluate_module()
    rows = [
        {"sentence": "flat result", "expected": "neutral"},
        {"sentence": "sales rose", "expected": "positive"},
        {"sentence": "profit fell", "expected": "negative"},
        {"sentence": "margin improved", "expected": "positive"},
    ]
    scores = [
        {"neutral": 3.0, "positive": 1.0, "negative": 0.0},
        {"neutral": 1.0, "positive": 2.0, "negative": 0.0},
        {"neutral": 0.0, "positive": 1.0, "negative": 2.0},
        {"neutral": 2.0, "positive": 1.6, "negative": 0.0},
    ]

    summary = evaluate_sentiment.summarize(rows, scores)

    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)
    assert summary["prediction_counts"] == {"neutral": 2, "positive": 1, "negative": 1}
    assert summary["confusion"]["positive"] == {"neutral": 1, "positive": 1, "negative": 0}

    biased_summary = evaluate_sentiment.summarize(rows, evaluate_sentiment.apply_bias(scores, 0.5, 0.0))
    assert biased_summary["accuracy"] == pytest.approx(1.0)
    assert biased_summary["macro_f1"] == pytest.approx(1.0)

    best = evaluate_sentiment.best_bias(rows, scores, max_bias=0.5, step=0.5)
    assert best["biases"] == {"positive": 0.5, "negative": 0.0}
    assert best["summary"]["macro_f1"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the evaluator")
def test_evaluate_sentiment_parse_choice_map_validates_labels():
    evaluate_sentiment = _load_evaluate_module()

    assert evaluate_sentiment.parse_choice_map("neutral=neutral,positive=up,negative=down") == {
        "neutral": "neutral",
        "positive": "up",
        "negative": "down",
    }

    with pytest.raises(ValueError, match="Missing labels"):
        evaluate_sentiment.parse_choice_map("neutral=neutral,positive=up")

    with pytest.raises(ValueError, match="Unknown label"):
        evaluate_sentiment.parse_choice_map("neutral=neutral,positive=up,other=down")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the evaluator")
def test_lightning_adapter_load_creates_temporary_single_process_group(monkeypatch, tmp_path):
    evaluate_sentiment = _load_evaluate_module()
    calls = []
    monkeypatch.setattr(evaluate_sentiment.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(evaluate_sentiment.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        evaluate_sentiment.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        evaluate_sentiment.torch.distributed,
        "destroy_process_group",
        lambda: calls.append(("destroy", None)),
    )

    with evaluate_sentiment._single_process_group(str(tmp_path)):
        calls.append(("body", None))

    assert [name for name, _ in calls] == ["init", "body", "destroy"]
    assert calls[0][1]["backend"] == "nccl"
    assert calls[0][1]["rank"] == 0
    assert calls[0][1]["world_size"] == 1
    assert calls[0][1]["init_method"].startswith("file://")


def test_lightning_reload_comparison_allows_bounded_loss_variation():
    assess_validation = _load_assess_module()
    validation = {
        "response_token_loss": 1.75,
        "response_token_count": 100,
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "confusion": {"neutral": {"neutral": 1}},
        "prediction_counts": {"neutral": 1},
    }
    first = {"validation": validation}
    second = {"validation": {**validation, "response_token_loss": 1.7504}}

    report = assess_validation.verify_reload_reproducibility(first, second)

    assert report["response_token_loss_delta"] == pytest.approx(4e-4)
    with pytest.raises(ValueError, match="response-token loss delta"):
        assess_validation.verify_reload_reproducibility(
            first,
            {"validation": {**validation, "response_token_loss": 1.751}},
        )
    with pytest.raises(ValueError, match="changed evaluation metrics"):
        assess_validation.verify_reload_reproducibility(
            first,
            {"validation": {**validation, "accuracy": 0.4}},
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to verify adapter tensors")
def test_lightning_evaluation_verifies_loaded_adapter_values():
    import torch

    evaluate_sentiment = _load_evaluate_module()
    incoming = OrderedDict(
        {
            "base_model.model.model.layers.0.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        }
    )
    loaded = {
        key.removeprefix(evaluate_sentiment.adapter_checkpoint.HF_PEFT_BASE_MODEL_PREFIX): value.to(torch.bfloat16)
        for key, value in incoming.items()
    }

    class Model:
        @staticmethod
        def state_dict():
            return {
                **loaded,
                "layers.0.weight": torch.ones(2, 2),
            }

    report = evaluate_sentiment._verify_loaded_adapter_state(Model(), incoming)

    assert report["loaded_tensor_count"] == 2
    assert report["loaded_matches_received_after_dtype_cast"] is True
    incoming["base_model.model.model.layers.0.lora_B.weight"][0, 0] = 8.0
    with pytest.raises(RuntimeError, match="reload changed 1 tensors"):
        evaluate_sentiment._verify_loaded_adapter_state(Model(), incoming)
