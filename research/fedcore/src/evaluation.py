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

"""Validation-only completion selection and held-out evaluation."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from src.metrics import binary_log_loss_sum, binary_metrics, safe_delta


@dataclass(frozen=True)
class PreparedSite:
    site: str
    labels: np.ndarray
    image_available: np.ndarray
    missing_logits: np.ndarray
    full_logits: np.ndarray
    predicted_delta: np.ndarray


def predict_delta(model: torch.nn.Module, payload: dict, batch_size: int = 256) -> np.ndarray:
    features = payload["missing_features"].float()
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), max(1, int(batch_size))):
            predictions.append(model(features[start : start + batch_size]).detach().cpu())
    return torch.cat(predictions).numpy() if predictions else np.empty(0, dtype=np.float32)


def prepare_site(site: str, payload: dict, model: torch.nn.Module) -> PreparedSite:
    return PreparedSite(
        site=site,
        labels=payload["labels"].detach().cpu().numpy().astype(np.int64),
        image_available=payload["image_available"].detach().cpu().numpy().astype(bool),
        missing_logits=payload["missing_logits"].detach().cpu().numpy().astype(np.float64),
        full_logits=payload["full_logits"].detach().cpu().numpy().astype(np.float64),
        predicted_delta=predict_delta(model, payload).astype(np.float64),
    )


def _policy_logits(site: PreparedSite, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    before = site.missing_logits.copy()
    before[site.image_available] = site.full_logits[site.image_available]
    after = before.copy()
    missing = ~site.image_available
    after[missing] = site.missing_logits[missing] + float(alpha) * site.predicted_delta[missing]
    return before, after


def select_alpha(
    sites: Iterable[PreparedSite], alpha_grid: Iterable[float], aggregate_loss_tolerance: float = 0.0
) -> tuple[float, list[dict[str, float | int | bool]]]:
    """Select using validation sufficient statistics only; alpha=0 must be present."""

    statistics = validation_sufficient_statistics(sites, alpha_grid)
    return select_alpha_from_statistics(statistics, aggregate_loss_tolerance)


def validation_sufficient_statistics(
    sites: Iterable[PreparedSite], alpha_grid: Iterable[float]
) -> list[dict[str, float | int | str]]:
    """Compute site-local loss sums and counts for validation selection."""

    sites = list(sites)
    alpha_grid = sorted({float(alpha) for alpha in alpha_grid})
    if 0.0 not in alpha_grid:
        raise ValueError("alpha_grid must include 0 for the exact identity fallback.")
    if not sites:
        raise ValueError("At least one validation site is required.")

    statistics = []
    for site in sites:
        for alpha in alpha_grid:
            _, after = _policy_logits(site, alpha)
            missing = ~site.image_available
            statistics.append(
                {
                    "site": site.site,
                    "alpha": alpha,
                    "missing_loss_sum": (
                        binary_log_loss_sum(after[missing], site.labels[missing]) if missing.any() else 0.0
                    ),
                    "missing_count": int(missing.sum()),
                    "aggregate_loss_sum": binary_log_loss_sum(after, site.labels),
                    "aggregate_count": int(site.labels.size),
                }
            )
    return statistics


def select_alpha_from_statistics(
    statistics: Iterable[dict[str, float | int | str]], aggregate_loss_tolerance: float = 0.0
) -> tuple[float, list[dict[str, float | int | bool]]]:
    """Aggregate site statistics and apply the validation no-harm constraint."""

    aggregate_loss_tolerance = float(aggregate_loss_tolerance)
    if not np.isfinite(aggregate_loss_tolerance) or aggregate_loss_tolerance < 0.0:
        raise ValueError("aggregate_loss_tolerance must be a finite, non-negative value.")
    statistics = list(statistics)
    alpha_grid = sorted({float(row["alpha"]) for row in statistics})
    if not statistics or 0.0 not in alpha_grid or not all(np.isfinite(alpha) for alpha in alpha_grid):
        raise ValueError("Validation statistics must be non-empty and include alpha=0.")

    rows = []
    for alpha in alpha_grid:
        alpha_rows = [row for row in statistics if float(row["alpha"]) == alpha]
        missing_loss_sum = sum(float(row["missing_loss_sum"]) for row in alpha_rows)
        missing_count = sum(int(row["missing_count"]) for row in alpha_rows)
        aggregate_loss_sum = sum(float(row["aggregate_loss_sum"]) for row in alpha_rows)
        aggregate_count = sum(int(row["aggregate_count"]) for row in alpha_rows)
        if missing_count == 0:
            raise ValueError("Validation data contain no naturally missing-image examples.")
        if aggregate_count <= 0:
            raise ValueError("Validation data contain no aggregate examples.")
        if not all(np.isfinite(value) for value in (missing_loss_sum, aggregate_loss_sum)):
            raise ValueError("Validation statistics contain non-finite loss values.")
        rows.append(
            {
                "alpha": alpha,
                "missing_count": missing_count,
                "missing_log_loss": missing_loss_sum / missing_count,
                "aggregate_count": aggregate_count,
                "aggregate_log_loss": aggregate_loss_sum / aggregate_count,
            }
        )

    identity_loss = next(float(row["aggregate_log_loss"]) for row in rows if float(row["alpha"]) == 0.0)
    for row in rows:
        row["feasible"] = float(row["aggregate_log_loss"]) <= identity_loss + aggregate_loss_tolerance + 1e-12
    feasible = [row for row in rows if bool(row["feasible"])]
    selected = min(feasible, key=lambda row: (float(row["missing_log_loss"]), abs(float(row["alpha"]))))
    return float(selected["alpha"]), rows


def _concatenate(sites: list[PreparedSite], field: str) -> np.ndarray:
    arrays = [np.asarray(getattr(site, field)) for site in sites]
    return np.concatenate(arrays) if arrays else np.empty(0)


def evaluate_sites(sites: Iterable[PreparedSite], alpha: float) -> tuple[dict, dict]:
    sites = list(sites)
    per_site = {}
    aggregate_labels = []
    aggregate_before = []
    aggregate_after = []
    natural_labels = []
    natural_before = []
    natural_after = []
    paired_labels = []
    paired_full = []
    paired_removed = []
    paired_completed = []

    for site in sites:
        before, after = _policy_logits(site, alpha)
        missing = ~site.image_available
        paired = site.image_available
        per_site[site.site] = {
            "image_available": int(paired.sum()),
            "image_missing": int(missing.sum()),
            "aggregate_before": binary_metrics(before, site.labels),
            "aggregate_after": binary_metrics(after, site.labels),
            "missing_before": binary_metrics(site.missing_logits[missing], site.labels[missing]),
            "missing_after": binary_metrics(after[missing], site.labels[missing]),
            "paired_full": binary_metrics(site.full_logits[paired], site.labels[paired]),
            "paired_removed": binary_metrics(site.missing_logits[paired], site.labels[paired]),
            "paired_completed": binary_metrics(
                site.missing_logits[paired] + float(alpha) * site.predicted_delta[paired], site.labels[paired]
            ),
        }
        aggregate_labels.append(site.labels)
        aggregate_before.append(before)
        aggregate_after.append(after)
        natural_labels.append(site.labels[missing])
        natural_before.append(site.missing_logits[missing])
        natural_after.append(after[missing])
        paired_labels.append(site.labels[paired])
        paired_full.append(site.full_logits[paired])
        paired_removed.append(site.missing_logits[paired])
        paired_completed.append(site.missing_logits[paired] + float(alpha) * site.predicted_delta[paired])

    labels = np.concatenate(aggregate_labels)
    missing_labels = np.concatenate(natural_labels)
    complete_labels = np.concatenate(paired_labels)
    aggregate_before_metrics = binary_metrics(np.concatenate(aggregate_before), labels)
    aggregate_after_metrics = binary_metrics(np.concatenate(aggregate_after), labels)
    missing_before_metrics = binary_metrics(np.concatenate(natural_before), missing_labels)
    missing_after_metrics = binary_metrics(np.concatenate(natural_after), missing_labels)
    paired_full_metrics = binary_metrics(np.concatenate(paired_full), complete_labels)
    paired_removed_metrics = binary_metrics(np.concatenate(paired_removed), complete_labels)
    paired_completed_metrics = binary_metrics(np.concatenate(paired_completed), complete_labels)

    missing_delta = safe_delta(float(missing_after_metrics["auroc"]), float(missing_before_metrics["auroc"]))
    aggregate_delta = safe_delta(float(aggregate_after_metrics["auroc"]), float(aggregate_before_metrics["auroc"]))
    paired_gap = safe_delta(float(paired_full_metrics["auroc"]), float(paired_removed_metrics["auroc"]))
    paired_gain = safe_delta(float(paired_completed_metrics["auroc"]), float(paired_removed_metrics["auroc"]))
    relative_lift = (
        100.0 * missing_delta / float(missing_before_metrics["auroc"])
        if float(missing_before_metrics["auroc"]) > 0.0
        else float("nan")
    )
    gap_recovered = 100.0 * paired_gain / paired_gap if paired_gap > 0.0 else float("nan")
    summary = {
        "selected_alpha": float(alpha),
        "missing_before": missing_before_metrics,
        "missing_after": missing_after_metrics,
        "missing_delta_auroc": missing_delta,
        "missing_relative_lift_pct": relative_lift,
        "aggregate_before": aggregate_before_metrics,
        "aggregate_after": aggregate_after_metrics,
        "aggregate_delta_auroc": aggregate_delta,
        "paired_full": paired_full_metrics,
        "paired_removed": paired_removed_metrics,
        "paired_completed": paired_completed_metrics,
        "paired_gap_recovered_pct": gap_recovered,
        "pooled_auroc_scope": "Simulator-only pooled evaluation; not a privacy-preserving production metric.",
    }
    return summary, per_site
