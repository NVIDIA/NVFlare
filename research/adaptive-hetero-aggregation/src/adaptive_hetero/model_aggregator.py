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

"""FLModel adapter for heterogeneity-aware aggregation in the unified FedAvg workflow."""

import math
from typing import Dict

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    WeightedAggregationHelper,
    filter_aggregatable_metrics,
)

from .nvflare_aggregator import AdaptiveMetaKey
from .policy import AdaptiveHeterogeneityPolicy, AdaptiveWeightingConfig


class AdaptiveHeterogeneityModelAggregator(ModelAggregator):
    """Apply the adaptive policy to ``FLModel`` weight-difference results.

    This adapter is for NVFlare's unified FedAvg workflow. It uses
    ``NUM_STEPS_CURRENT_ROUND`` only for the native FedAvg fallback weight and
    uses ``AdaptiveMetaKey.SAMPLE_COUNT`` for the policy's sample-based terms.
    Per-site final weights remain server-side in ``last_weights`` and are not
    copied into the returned global model metadata.
    """

    expected_data_kind = None

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
        self._results: Dict[str, FLModel] = {}
        self.last_weights: Dict[str, float] = {}

    def reset_stats(self):
        self._results = {}

    @staticmethod
    def _client_name(model: FLModel) -> str:
        value = (model.meta or {}).get("client_name")
        if not isinstance(value, str) or not value:
            raise ValueError("adaptive client result is missing FLModel.meta['client_name']")
        return value

    @staticmethod
    def _finite_positive(meta: dict, key: str, client_name: str) -> float:
        try:
            value = float(meta[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"client {client_name!r} is missing valid {key!r} metadata") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"client {client_name!r} must provide finite positive {key!r} metadata")
        return value

    def accept_model(self, model: FLModel):
        client_name = self._client_name(model)
        if model.params_type != ParamsType.DIFF:
            raise ValueError(
                f"adaptive aggregation requires ParamsType.DIFF, got {model.params_type} from client {client_name!r}"
            )
        if not model.params:
            raise ValueError(f"adaptive aggregation received empty parameters from client {client_name!r}")
        if client_name in self._results:
            raise ValueError(f"adaptive aggregation received duplicate result from client {client_name!r}")

        meta = model.meta or {}
        self._finite_positive(meta, FLMetaKey.NUM_STEPS_CURRENT_ROUND, client_name)
        self._finite_positive(meta, AdaptiveMetaKey.SAMPLE_COUNT, client_name)

        descriptor = meta.get(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR)
        try:
            descriptor = [float(value) for value in descriptor]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"client {client_name!r} is missing valid {AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR!r} metadata"
            ) from exc
        if not descriptor or any(not math.isfinite(value) or value < 0.0 for value in descriptor):
            raise ValueError(f"client {client_name!r} provided an invalid distribution descriptor")
        if sum(descriptor) <= 0.0:
            raise ValueError(f"client {client_name!r} distribution descriptor must have positive mass")

        try:
            metric = float(meta[AdaptiveMetaKey.CLIENT_METRIC])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"client {client_name!r} is missing valid {AdaptiveMetaKey.CLIENT_METRIC!r} metadata"
            ) from exc
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(f"client {client_name!r} metric must be a finite value in [0, 1]")

        quality = meta.get(AdaptiveMetaKey.QUALITY_IMPROVEMENT)
        if quality is not None:
            try:
                quality = float(quality)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"client {client_name!r} provided invalid quality metadata") from exc
            if not math.isfinite(quality):
                raise ValueError(f"client {client_name!r} quality metadata must be finite")
        if self.quality_exponent > 0.0 and quality is None:
            raise ValueError(f"client {client_name!r} must provide quality metadata when quality weighting is enabled")

        # Store normalized validated values back into the local result only. This
        # object remains server-side and is discarded after aggregation.
        meta[AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR] = descriptor
        meta[AdaptiveMetaKey.CLIENT_METRIC] = metric
        if quality is not None:
            meta[AdaptiveMetaKey.QUALITY_IMPROVEMENT] = quality
        model.meta = meta
        self._results[client_name] = model

    def aggregate_model(self) -> FLModel:
        if not self._results:
            # The unified FedAvg controller expects an FLModel from a custom
            # ModelAggregator. An empty DIFF is a safe no-op update and avoids
            # panicking the job when no client result is available for a round.
            self.last_weights = {}
            return FLModel(
                params={},
                params_type=ParamsType.DIFF,
                metrics=None,
                meta={
                    "nr_aggregated": 0,
                    "adaptive_empty_result": True,
                },
            )

        clients = sorted(self._results)
        results = [self._results[client] for client in clients]
        steps = np.asarray(
            [float(result.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND]) for result in results], dtype=np.float64
        )
        native_weights = steps / steps.sum()
        qualities = [result.meta.get(AdaptiveMetaKey.QUALITY_IMPROVEMENT) for result in results]
        quality_values = None if self.quality_exponent == 0.0 else qualities

        policy_result = self.policy.compute(
            sample_counts=[float(result.meta[AdaptiveMetaKey.SAMPLE_COUNT]) for result in results],
            descriptors=[result.meta[AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR] for result in results],
            client_metrics=[float(result.meta[AdaptiveMetaKey.CLIENT_METRIC]) for result in results],
            base_weights=native_weights,
            quality_improvements=quality_values,
            cohort_key=tuple(clients),
        )

        params_helper = WeightedAggregationHelper()
        metrics_helper = WeightedAggregationHelper()
        all_metrics = True
        current_round = None
        for client, result, weight in zip(clients, results, policy_result.weights):
            current_round = result.current_round
            params_helper.add(
                data=result.params,
                weight=float(weight),
                contributor_name=client,
                contribution_round=current_round,
            )
            if result.metrics is None:
                all_metrics = False
            elif all_metrics:
                aggregatable_metrics = filter_aggregatable_metrics(result.metrics)
                if aggregatable_metrics:
                    metrics_helper.add(
                        data=aggregatable_metrics,
                        weight=float(weight),
                        contributor_name=client,
                        contribution_round=current_round,
                    )

        self.last_weights = {client: float(weight) for client, weight in zip(clients, policy_result.weights)}
        metrics = metrics_helper.get_result() if all_metrics and metrics_helper.total else None
        aggregated = FLModel(
            params=params_helper.get_result(),
            params_type=ParamsType.DIFF,
            metrics=metrics or None,
            current_round=current_round,
            meta={
                AdaptiveMetaKey.MEAN_HETEROGENEITY: policy_result.mean_heterogeneity,
                AdaptiveMetaKey.RAW_METRIC_GAP: policy_result.raw_metric_gap,
                AdaptiveMetaKey.METRIC_GAP: policy_result.metric_gap,
                AdaptiveMetaKey.PERFORMANCE_GATE: policy_result.performance_gate,
                AdaptiveMetaKey.CANDIDATE_BLEND_FACTOR: policy_result.candidate_blend_factor,
                AdaptiveMetaKey.BLEND_FACTOR: policy_result.blend_factor,
                AdaptiveMetaKey.ACTIVATION_STREAK: policy_result.activation_streak,
                AdaptiveMetaKey.BOUNDS_FEASIBLE: policy_result.bounds_feasible,
                "nr_aggregated": len(clients),
            },
        )
        self.reset_stats()
        return aggregated
