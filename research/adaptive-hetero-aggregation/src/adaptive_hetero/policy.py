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

"""Conservative adaptive weighting for heterogeneous federated learning."""

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveWeightingConfig:
    """Configuration for :class:`AdaptiveHeterogeneityPolicy`.

    Client metrics must be normalized higher-is-better scores in ``[0, 1]``.
    Small-client metrics are shrunk toward the sample-weighted federation mean
    before they are used for fairness pressure. Adaptive weighting also needs
    sustained evidence for a stable participating cohort.
    """

    sample_exponent: float = 0.65
    representation_exponent: float = 0.70
    quality_exponent: float = 0.0
    fairness_strength: float = 1.0
    metric_prior_strength: float = 100.0
    heterogeneity_threshold: float = 0.26
    heterogeneity_temperature: float = 0.04
    heterogeneity_deadband: float = 0.15
    performance_gap_threshold: float = 0.10
    performance_gap_temperature: float = 0.03
    performance_gap_deadband: float = 0.05
    max_blend_factor: float = 0.20
    activation_warmup_rounds: int = 3
    activation_patience: int = 2
    require_stable_cohort: bool = True
    min_weight: float = 0.02
    max_weight: float = 0.50
    epsilon: float = 1e-12


@dataclass(frozen=True)
class WeightingResult:
    weights: np.ndarray
    base_weights: np.ndarray
    adaptive_weights: np.ndarray
    client_js_divergence: np.ndarray
    representation_scores: np.ndarray
    quality_scores: np.ndarray
    fairness_scores: np.ndarray
    adjusted_metrics: np.ndarray
    mean_heterogeneity: float
    raw_metric_gap: float
    metric_gap: float
    heterogeneity_gate: float
    performance_gate: float
    candidate_blend_factor: float
    blend_factor: float
    activation_streak: int


def _normalize_distribution(values: Sequence[float], epsilon: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("distribution descriptors must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("distribution descriptors must contain finite non-negative values")
    total = float(array.sum())
    if total <= epsilon:
        raise ValueError("distribution descriptor must have positive mass")
    array = np.clip(array / total, epsilon, None)
    return array / array.sum()


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def project_bounded_simplex(values: Sequence[float], lower: float, upper: float) -> np.ndarray:
    """Project values onto ``sum(w)=1`` with element-wise lower/upper bounds."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    count = values.size
    if lower < 0.0 or upper <= 0.0 or lower > upper:
        raise ValueError("invalid weight bounds")
    if lower * count > 1.0 + 1e-12 or upper * count < 1.0 - 1e-12:
        raise ValueError("weight bounds are infeasible for the number of clients")

    lo_tau = float(np.min(values - upper))
    hi_tau = float(np.max(values - lower))
    for _ in range(100):
        tau = 0.5 * (lo_tau + hi_tau)
        projected = np.clip(values - tau, lower, upper)
        if projected.sum() > 1.0:
            lo_tau = tau
        else:
            hi_tau = tau

    projected = np.clip(values - 0.5 * (lo_tau + hi_tau), lower, upper)
    residual = 1.0 - float(projected.sum())
    tolerance = 1e-12
    if abs(residual) > tolerance:
        room = upper - projected if residual > 0.0 else projected - lower
        direction = 1.0 if residual > 0.0 else -1.0
        remaining = abs(residual)
        for index in np.argsort(room)[::-1]:
            if remaining <= tolerance:
                break
            delta = min(remaining, float(room[index]))
            projected[index] += direction * delta
            remaining -= delta
        if remaining > 1e-10:
            raise RuntimeError("failed to satisfy bounded-simplex constraints numerically")

    if not np.isclose(projected.sum(), 1.0, atol=1e-10):
        raise RuntimeError("bounded-simplex projection does not sum to one")
    if np.any(projected < lower - 1e-10) or np.any(projected > upper + 1e-10):
        raise RuntimeError("bounded-simplex projection violated configured bounds")
    return projected


class AdaptiveHeterogeneityPolicy:
    """Compute adaptive weights only after stable, sustained evidence."""

    def __init__(self, config: AdaptiveWeightingConfig | None = None):
        self.config = config or AdaptiveWeightingConfig()
        self._validate_config()
        self.reset_state()

    def reset_state(self):
        self._rounds_seen = 0
        self._activation_streak = 0
        self._last_cohort_key: Hashable | None = None

    def _validate_config(self):
        cfg = self.config
        if not 0.0 < cfg.sample_exponent <= 1.0:
            raise ValueError("sample_exponent must be in (0, 1]")
        if cfg.representation_exponent < 0.0 or cfg.quality_exponent < 0.0:
            raise ValueError("representation/quality exponents must be non-negative")
        if cfg.fairness_strength < 0.0 or cfg.metric_prior_strength < 0.0:
            raise ValueError("fairness_strength and metric_prior_strength must be non-negative")
        if cfg.heterogeneity_temperature <= 0.0 or cfg.performance_gap_temperature <= 0.0:
            raise ValueError("gate temperatures must be positive")
        if cfg.heterogeneity_deadband < 0.0 or cfg.performance_gap_deadband < 0.0:
            raise ValueError("deadbands must be non-negative")
        if not 0.0 < cfg.max_blend_factor <= 1.0:
            raise ValueError("max_blend_factor must be in (0, 1]")
        if cfg.activation_warmup_rounds < 0 or cfg.activation_patience < 1:
            raise ValueError("activation warmup/patience are invalid")
        if cfg.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if cfg.min_weight < 0.0 or cfg.max_weight <= 0.0 or cfg.min_weight > cfg.max_weight:
            raise ValueError("invalid weight bounds")

    def compute(
        self,
        sample_counts: Sequence[float],
        descriptors: Sequence[Sequence[float]],
        client_metrics: Sequence[float],
        quality_improvements: Sequence[float] | None = None,
        cohort_key: Hashable | None = None,
    ) -> WeightingResult:
        cfg = self.config
        counts = np.asarray(sample_counts, dtype=np.float64)
        metrics = np.asarray(client_metrics, dtype=np.float64)
        if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0.0) or not np.all(np.isfinite(counts)):
            raise ValueError("sample_counts must contain finite positive values")
        if metrics.shape != counts.shape or not np.all(np.isfinite(metrics)):
            raise ValueError("client_metrics must be finite and match sample_counts")
        if np.any(metrics < 0.0) or np.any(metrics > 1.0):
            raise ValueError("client_metrics must be normalized higher-is-better values in [0, 1]")
        if len(descriptors) != counts.size:
            raise ValueError("descriptors must contain one descriptor per client")

        normalized_descriptors = [_normalize_distribution(item, cfg.epsilon) for item in descriptors]
        if len({item.size for item in normalized_descriptors}) != 1:
            raise ValueError("all descriptors must have the same length")
        descriptor_matrix = np.stack(normalized_descriptors)

        base_weights = counts / counts.sum()
        reference = _normalize_distribution(np.average(descriptor_matrix, axis=0, weights=counts), cfg.epsilon)
        js_values = np.asarray([_jensen_shannon(item, reference) for item in descriptor_matrix], dtype=np.float64)
        mean_heterogeneity = float(np.average(js_values, weights=counts))

        novelty = js_values / max(float(js_values.max()), cfg.epsilon)
        inverse_frequency = 1.0 / np.sqrt(reference + cfg.epsilon)
        inverse_frequency /= float(np.dot(reference, inverse_frequency))
        rarity = descriptor_matrix @ inverse_frequency
        rarity /= max(float(rarity.max()), cfg.epsilon)
        representation_scores = np.clip(0.5 + 0.9 * novelty + 0.1 * rarity, 0.5, 1.5)

        metric_mean = float(np.average(metrics, weights=counts))
        reliability = counts / (counts + cfg.metric_prior_strength)
        adjusted_metrics = reliability * metrics + (1.0 - reliability) * metric_mean
        raw_metric_gap = float(metrics.max() - metrics.min())
        metric_gap = float(adjusted_metrics.max() - adjusted_metrics.min())
        absolute_deficit = np.clip(float(adjusted_metrics.max()) - adjusted_metrics, 0.0, 1.0)
        fairness_scores = np.exp(cfg.fairness_strength * absolute_deficit)

        if cfg.quality_exponent == 0.0 or quality_improvements is None:
            quality = np.ones_like(counts)
        else:
            quality = np.asarray(quality_improvements, dtype=np.float64)
            if quality.shape != counts.shape or not np.all(np.isfinite(quality)):
                raise ValueError("quality_improvements must be finite and match sample_counts")
            quality_span = float(quality.max() - quality.min())
            if quality_span <= cfg.epsilon:
                quality = np.ones_like(counts)
            else:
                quality = 0.75 + 0.50 * (quality - float(quality.min())) / quality_span

        raw = (
            np.power(counts, cfg.sample_exponent)
            * np.power(representation_scores, cfg.representation_exponent)
            * fairness_scores
            * np.power(quality, cfg.quality_exponent)
        )
        raw /= raw.sum()
        adaptive_weights = project_bounded_simplex(raw, cfg.min_weight, cfg.max_weight)

        heterogeneity_gate = (
            0.0
            if mean_heterogeneity <= cfg.heterogeneity_deadband
            else _sigmoid((mean_heterogeneity - cfg.heterogeneity_threshold) / cfg.heterogeneity_temperature)
        )
        performance_gate = (
            0.0
            if metric_gap <= cfg.performance_gap_deadband
            else _sigmoid((metric_gap - cfg.performance_gap_threshold) / cfg.performance_gap_temperature)
        )
        candidate_blend = cfg.max_blend_factor * heterogeneity_gate * performance_gate

        self._rounds_seen += 1
        if cfg.require_stable_cohort and self._last_cohort_key is not None and cohort_key != self._last_cohort_key:
            self._activation_streak = 0
        self._last_cohort_key = cohort_key
        if candidate_blend > 0.0:
            self._activation_streak += 1
        else:
            self._activation_streak = 0

        blend = candidate_blend
        if self._rounds_seen <= cfg.activation_warmup_rounds or self._activation_streak < cfg.activation_patience:
            blend = 0.0

        weights = base_weights.copy() if blend == 0.0 else (1.0 - blend) * base_weights + blend * adaptive_weights
        if blend != 0.0:
            weights /= weights.sum()
        return WeightingResult(
            weights=weights,
            base_weights=base_weights,
            adaptive_weights=adaptive_weights,
            client_js_divergence=js_values,
            representation_scores=representation_scores,
            quality_scores=quality,
            fairness_scores=fairness_scores,
            adjusted_metrics=adjusted_metrics,
            mean_heterogeneity=mean_heterogeneity,
            raw_metric_gap=raw_metric_gap,
            metric_gap=metric_gap,
            heterogeneity_gate=float(heterogeneity_gate),
            performance_gate=float(performance_gate),
            candidate_blend_factor=float(candidate_blend),
            blend_factor=float(blend),
            activation_streak=self._activation_streak,
        )
