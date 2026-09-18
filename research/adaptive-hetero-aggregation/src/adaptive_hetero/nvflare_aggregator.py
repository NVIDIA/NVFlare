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

"""NVFlare ``Aggregator`` adapter for adaptive heterogeneity-aware weighting."""

import math
from dataclasses import dataclass

import numpy as np
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants

from .policy import AdaptiveHeterogeneityPolicy, AdaptiveWeightingConfig


class AdaptiveMetaKey:
    """Client metadata keys consumed or produced by the research aggregator."""

    DISTRIBUTION_DESCRIPTOR = "adaptive_distribution_descriptor"
    CLIENT_METRIC = "adaptive_client_metric"
    SAMPLE_COUNT = "adaptive_sample_count"
    QUALITY_IMPROVEMENT = "adaptive_quality_improvement"
    MEAN_HETEROGENEITY = "adaptive_mean_heterogeneity"
    RAW_METRIC_GAP = "adaptive_raw_metric_gap"
    METRIC_GAP = "adaptive_metric_gap"
    PERFORMANCE_GATE = "adaptive_performance_gate"
    CANDIDATE_BLEND_FACTOR = "adaptive_candidate_blend_factor"
    BLEND_FACTOR = "adaptive_blend_factor"
    ACTIVATION_STREAK = "adaptive_activation_streak"
    BOUNDS_FEASIBLE = "adaptive_bounds_feasible"
    AGGREGATION_ROUNDS = "adaptive_aggregation_rounds"
    ACTIVE_ROUNDS = "adaptive_active_rounds"
    ACTIVATION_RATE = "adaptive_activation_rate"
    MEAN_ACTIVE_BLEND_FACTOR = "adaptive_mean_active_blend_factor"
    MAX_OBSERVED_BLEND_FACTOR = "adaptive_max_observed_blend_factor"
    COHORT_CHANGE_COUNT = "adaptive_cohort_change_count"


@dataclass
class _Contribution:
    name: str
    round_number: int
    data: dict
    local_steps: float
    sample_count: float
    descriptor: list[float]
    metric: float
    quality_improvement: float | None


class AdaptiveHeterogeneityAggregator(Aggregator):
    """Aggregate ``WEIGHT_DIFF`` updates using conservative adaptive weights.

    The native fallback uses ``NUM_STEPS_CURRENT_ROUND`` exactly as NVFlare's
    standard weighted aggregator does. The adaptive policy receives a separate
    actual training-example count through ``AdaptiveMetaKey.SAMPLE_COUNT`` so
    sample-based reliability and representation calculations do not confuse
    optimizer steps with examples.

    ``CLIENT_METRIC`` must be a normalized higher-is-better value in ``[0, 1]``.
    Quality weighting is disabled by default because local quality-improvement
    values are not automatically comparable across sites.
    """

    expected_data_kind = DataKind.WEIGHT_DIFF

    def __init__(
        self,
        sample_exponent: float = 0.65,
        representation_exponent: float = 0.70,
        quality_exponent: float = 0.0,
        fairness_strength: float = 1.0,
        metric_prior_strength: float = 100.0,
        heterogeneity_threshold: float = 0.26,
        heterogeneity_temperature: float = 0.04,
        heterogeneity_deadband: float = 0.15,
        performance_gap_threshold: float = 0.10,
        performance_gap_temperature: float = 0.03,
        performance_gap_deadband: float = 0.05,
        max_blend_factor: float = 0.20,
        activation_warmup_rounds: int = 3,
        activation_patience: int = 2,
        require_stable_cohort: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        super().__init__()

        # Keep constructor arguments as public attributes. NVFlare's FedJob
        # component serializer reconstructs custom components from constructor
        # arguments, so hiding them only inside a nested config loses non-default
        # values when the job is exported to the server.
        self.sample_exponent = sample_exponent
        self.representation_exponent = representation_exponent
        self.quality_exponent = quality_exponent
        self.fairness_strength = fairness_strength
        self.metric_prior_strength = metric_prior_strength
        self.heterogeneity_threshold = heterogeneity_threshold
        self.heterogeneity_temperature = heterogeneity_temperature
        self.heterogeneity_deadband = heterogeneity_deadband
        self.performance_gap_threshold = performance_gap_threshold
        self.performance_gap_temperature = performance_gap_temperature
        self.performance_gap_deadband = performance_gap_deadband
        self.max_blend_factor = max_blend_factor
        self.activation_warmup_rounds = activation_warmup_rounds
        self.activation_patience = activation_patience
        self.require_stable_cohort = require_stable_cohort
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.config = AdaptiveWeightingConfig(
            sample_exponent=self.sample_exponent,
            representation_exponent=self.representation_exponent,
            quality_exponent=self.quality_exponent,
            fairness_strength=self.fairness_strength,
            metric_prior_strength=self.metric_prior_strength,
            heterogeneity_threshold=self.heterogeneity_threshold,
            heterogeneity_temperature=self.heterogeneity_temperature,
            heterogeneity_deadband=self.heterogeneity_deadband,
            performance_gap_threshold=self.performance_gap_threshold,
            performance_gap_temperature=self.performance_gap_temperature,
            performance_gap_deadband=self.performance_gap_deadband,
            max_blend_factor=self.max_blend_factor,
            activation_warmup_rounds=self.activation_warmup_rounds,
            activation_patience=self.activation_patience,
            require_stable_cohort=self.require_stable_cohort,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )
        self.policy = AdaptiveHeterogeneityPolicy(self.config)
        self._contributions: dict[str, _Contribution] = {}
        self._processed_algorithm = None
        self._descriptor_size = None
        self.last_weights: dict[str, float] = {}

    def reset(self, fl_ctx: FLContext):
        """Clear per-round contributions while preserving activation history."""
        self._contributions = {}
        self._processed_algorithm = None
        self._descriptor_size = None

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind != DataKind.WEIGHT_DIFF or not dxo.data:
            self.log_error(fl_ctx, "expected a non-empty WEIGHT_DIFF payload")
            return False

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if contribution_round != current_round:
            self.log_warning(fl_ctx, f"discarding contribution from {contributor_name!r} for wrong round")
            return False
        if contributor_name in self._contributions:
            self.log_warning(fl_ctx, f"discarding duplicate contribution from {contributor_name!r}")
            return False
        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(fl_ctx, f"contributor {contributor_name!r} returned rc={rc}")
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            if self._processed_algorithm is None:
                self._processed_algorithm = processed_algorithm
            elif self._processed_algorithm != processed_algorithm:
                self.log_error(fl_ctx, "all updates must use the same processed algorithm")
                return False

        local_steps = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
        sample_count = dxo.get_meta_prop(AdaptiveMetaKey.SAMPLE_COUNT)
        descriptor = dxo.get_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR)
        metric = dxo.get_meta_prop(AdaptiveMetaKey.CLIENT_METRIC)
        quality = dxo.get_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, None)
        try:
            local_steps = float(local_steps)
            sample_count = float(sample_count)
            metric = float(metric)
            descriptor = [float(value) for value in descriptor]
            quality = None if quality is None else float(quality)
        except (TypeError, ValueError):
            self.log_error(fl_ctx, f"contributor {contributor_name!r} is missing valid adaptive metadata")
            return False

        if (
            local_steps <= 0.0
            or not math.isfinite(local_steps)
            or sample_count <= 0.0
            or not math.isfinite(sample_count)
            or not math.isfinite(metric)
            or metric < 0.0
            or metric > 1.0
            or not descriptor
            or any(not math.isfinite(value) or value < 0.0 for value in descriptor)
            or sum(descriptor) <= 0.0
            or (quality is not None and not math.isfinite(quality))
        ):
            self.log_error(fl_ctx, f"invalid adaptive aggregation metadata from {contributor_name!r}")
            return False

        if self.config.quality_exponent > 0.0 and quality is None:
            self.log_error(
                fl_ctx, f"quality metadata is required for {contributor_name!r} when quality weighting is enabled"
            )
            return False

        if self._descriptor_size is None:
            self._descriptor_size = len(descriptor)
        elif len(descriptor) != self._descriptor_size:
            self.log_error(fl_ctx, f"descriptor size mismatch from {contributor_name!r}")
            return False

        self._contributions[contributor_name] = _Contribution(
            name=contributor_name,
            round_number=int(contribution_round),
            data=dxo.data,
            local_steps=local_steps,
            sample_count=sample_count,
            descriptor=descriptor,
            metric=metric,
            quality_improvement=quality,
        )
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if not self._contributions:
            self.log_warning(fl_ctx, "no valid contributions were accepted for this aggregation round")
            self.last_weights = {}
            self.reset(fl_ctx)
            return make_reply(ReturnCode.EMPTY_RESULT)

        names = sorted(self._contributions)
        contributions = [self._contributions[name] for name in names]
        qualities = [item.quality_improvement for item in contributions]
        quality_values = None if self.config.quality_exponent == 0.0 else qualities
        local_steps = np.asarray([item.local_steps for item in contributions], dtype=np.float64)
        native_weights = local_steps / local_steps.sum()
        result = self.policy.compute(
            sample_counts=[item.sample_count for item in contributions],
            descriptors=[item.descriptor for item in contributions],
            client_metrics=[item.metric for item in contributions],
            base_weights=native_weights,
            quality_improvements=quality_values,
            cohort_key=tuple(names),
        )

        helper = WeightedAggregationHelper()
        for item, weight in zip(contributions, result.weights):
            helper.add(
                data=item.data,
                weight=float(weight),
                contributor_name=item.name,
                contribution_round=item.round_number,
            )

        # Per-site weights are intentionally kept server-side. Putting the table
        # in aggregated DXO metadata would copy it into the global model and
        # broadcast every participant's weight to all clients.
        self.last_weights = {name: float(weight) for name, weight in zip(names, result.weights)}
        dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=helper.get_result(),
            meta={
                AdaptiveMetaKey.MEAN_HETEROGENEITY: result.mean_heterogeneity,
                AdaptiveMetaKey.RAW_METRIC_GAP: result.raw_metric_gap,
                AdaptiveMetaKey.METRIC_GAP: result.metric_gap,
                AdaptiveMetaKey.PERFORMANCE_GATE: result.performance_gate,
                AdaptiveMetaKey.CANDIDATE_BLEND_FACTOR: result.candidate_blend_factor,
                AdaptiveMetaKey.BLEND_FACTOR: result.blend_factor,
                AdaptiveMetaKey.ACTIVATION_STREAK: result.activation_streak,
                AdaptiveMetaKey.BOUNDS_FEASIBLE: result.bounds_feasible,
            },
        )
        if self._processed_algorithm is not None:
            dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, self._processed_algorithm)
        self.reset(fl_ctx)
        return dxo.to_shareable()
