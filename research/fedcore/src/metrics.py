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

"""Dependency-light binary classification metrics for the starter."""

import math

import numpy as np


def binary_log_loss_sum(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must have the same shape, got {logits.shape} and {labels.shape}")
    return float(np.logaddexp(0.0, logits).sum() - np.multiply(labels, logits).sum())


def binary_auroc(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC with average ranks for ties."""

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = float(ranks[targets == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != targets.shape:
        raise ValueError(f"logits and labels must have the same shape, got {scores.shape} and {targets.shape}")
    count = int(targets.size)
    if count == 0:
        return {"count": 0, "positive": 0, "accuracy": float("nan"), "log_loss": float("nan"), "auroc": float("nan")}
    predictions = (scores >= 0.0).astype(np.int64)
    return {
        "count": count,
        "positive": int(targets.sum()),
        "accuracy": float((predictions == targets).mean()),
        "log_loss": binary_log_loss_sum(scores, targets) / count,
        "auroc": binary_auroc(scores, targets),
    }


def safe_delta(after: float, before: float) -> float:
    if math.isnan(after) or math.isnan(before):
        return float("nan")
    return after - before
