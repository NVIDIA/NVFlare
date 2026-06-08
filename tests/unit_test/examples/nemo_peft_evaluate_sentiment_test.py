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
