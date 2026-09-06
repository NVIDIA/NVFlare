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

"""Adaptive weighting policy for heterogeneous federated learning.

The policy separates *when* heterogeneity-aware weighting should matter from
*how* client influence is redistributed. A deadband provides an exact ordinary
sample-weighting fallback when heterogeneity is small. Above that deadband, a
capped sigmoid gradually enables bounded weights combining sub-linear client
volume, representation benefit, client quality, and minimax-style pressure.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveWeightingConfig:
    """Configuration for :class:`AdaptiveHeterogeneityPolicy`."""

    sample_exponent: float = 0.65
    representation_exponent: float = 0.70
    quality_exponent: float = 0.40
    fairness_strength: float = 1.00
    heterogeneity_threshold: float = 0.26
    heterogeneity_temperature: float = 0.04
    heterogeneity_deadband: float = 0.15
    max_blend_factor: float = 0.40
    min_weight: float = 0.02
    max_weight: float = 0.50
    epsilon: float = 1e-12


@dataclass(frozen=True)
class WeightingResult:
    """Diagnostics and final normalized client weights."""

    weights: np.ndarray
    base_weights: np.ndarray
    adaptive_weights: np.ndarray
    client_js_divergence: np.ndarray
    representation_scores: np.ndarray
    quality_scores: np.ndarray
    fairness_scores: np.ndarray
    mean_heterogeneity: float
    blend_factor: float


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


def project_bounded_simplex(values: Sequence[float], lower: float, upper: float) -> np.ndarray:
    """Project values onto ``sum(w)=1`` with lower/upper element-wise bounds.

    The Euclidean projection has form ``clip(values - tau, lower, upper)``. A
    monotone bisection finds ``tau``. Any final floating-point residual is
    redistributed only into coordinates that still have room rather than
    renormalizing all coordinates and risking a bound violation.
    """

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
        if residual > 0.0:
            room = upper - projected
            direction = 1.0
        else:
            room = projected - lower
            direction = -1.0
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
    """Compute adaptive aggregation weights from client-level metadata.

    ``client_metrics`` are higher-is-better metrics. ``quality_improvements``
    should be comparable within a round and can represent validation-loss
    improvement of the local update over the received global model.
    """

    def __init__(self, config: AdaptiveWeightingConfig | None = None):
        self.config = config or AdaptiveWeightingConfig()
        self._validate_config()

    def _validate_config(self):
        cfg = self.config
        if not 0.0 < cfg.sample_exponent <= 1.0:
            raise ValueError("sample_exponent must be in (0, 1]")
        if cfg.representation_exponent < 0.0 or cfg.quality_exponent < 0.0:
            raise ValueError("representation/quality exponents must be non-negative")
        if cfg.fairness_strength < 0.0:
            raise ValueError("fairness_strength must be non-negative")
        if cfg.heterogeneity_temperature <= 0.0:
            raise ValueError("heterogeneity_temperature must be positive")
        if cfg.heterogeneity_deadband < 0.0:
            raise ValueError("heterogeneity_deadband must be non-negative")
        if not 0.0 < cfg.max_blend_factor <= 1.0:
            raise ValueError("max_blend_factor must be in (0, 1]")
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
    ) -> WeightingResult:
        cfg = self.config
        counts = np.asarray(sample_counts, dtype=np.float64)
        metrics = np.asarray(client_metrics, dtype=np.float64)
        if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0.0) or not np.all(np.isfinite(counts)):
            raise ValueError("sample_counts must contain finite positive values")
        if metrics.shape != counts.shape or not np.all(np.isfinite(metrics)):
            raise ValueError("client_metrics must be finite and match sample_counts")
        if len(descriptors) != counts.size:
            raise ValueError("descriptors must contain one descriptor per client")

        normalized_descriptors = [_normalize_distribution(item, cfg.epsilon) for item in descriptors]
        if len({item.size for item in normalized_descriptors}) != 1:
            raise ValueError("all descriptors must have the same length")
        descriptor_matrix = np.stack(normalized_descriptors)

        base_weights = counts / counts.sum()
        reference = np.average(descriptor_matrix, axis=0, weights=counts)
        reference = _normalize_distribution(reference, cfg.epsilon)
        js_values = np.asarray([_jensen_shannon(item, reference) for item in descriptor_matrix], dtype=np.float64)
        mean_heterogeneity = float(np.average(js_values, weights=counts))

        # Divergence alone is not treated as usefulness. Rarity rewards mass on
        # globally underrepresented descriptor bins while novelty captures shift.
        novelty = js_values / max(float(js_values.max()), cfg.epsilon)
        inverse_frequency = 1.0 / np.sqrt(reference + cfg.epsilon)
        inverse_frequency /= float(np.dot(reference, inverse_frequency))
        rarity = descriptor_matrix @ inverse_frequency
        rarity = rarity / max(float(rarity.max()), cfg.epsilon)
        representation_scores = np.clip(0.5 + 0.9 * novelty + 0.1 * rarity, 0.5, 1.5)

        metric_span = max(float(metrics.max() - metrics.min()), cfg.epsilon)
        deficit = (float(metrics.max()) - metrics) / metric_span
        fairness_scores = np.exp(cfg.fairness_strength * deficit)

        if quality_improvements is None:
            quality = np.ones_like(counts)
        else:
            quality = np.asarray(quality_improvements, dtype=np.float64)
            if quality.shape != counts.shape or not np.all(np.isfinite(quality)):
                raise ValueError("quality_improvements must be finite and match sample_counts")
            quality_span = max(float(quality.max() - quality.min()), cfg.epsilon)
            scaled_quality = (quality - float(quality.min())) / quality_span
            quality = 0.75 + 0.50 * scaled_quality

        raw = (
            np.power(counts, cfg.sample_exponent)
            * np.power(representation_scores, cfg.representation_exponent)
            * fairness_scores
            * np.power(quality, cfg.quality_exponent)
        )
        raw /= raw.sum()
        adaptive_weights = project_bounded_simplex(raw, cfg.min_weight, cfg.max_weight)

        # Below the deadband, preserve the exact ordinary sample-weighting path.
        # Above it, keep a smooth sigmoid response but cap the intervention so
        # noisy client-local metrics cannot completely dominate aggregation.
        if mean_heterogeneity <= cfg.heterogeneity_deadband:
            blend = 0.0
        else:
            z = (mean_heterogeneity - cfg.heterogeneity_threshold) / cfg.heterogeneity_temperature
            if z >= 0.0:
                blend = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                blend = exp_z / (1.0 + exp_z)
            blend = min(blend, cfg.max_blend_factor)

        weights = (1.0 - blend) * base_weights + blend * adaptive_weights
        weights /= weights.sum()
        return WeightingResult(
            weights=weights,
            base_weights=base_weights,
            adaptive_weights=adaptive_weights,
            client_js_divergence=js_values,
            representation_scores=representation_scores,
            quality_scores=quality,
            fairness_scores=fairness_scores,
            mean_heterogeneity=mean_heterogeneity,
            blend_factor=float(blend),
        )
