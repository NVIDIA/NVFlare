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

from dataclasses import dataclass
import math

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants

from .policy import AdaptiveHeterogeneityPolicy, AdaptiveWeightingConfig


class AdaptiveMetaKey:
    """Client metadata keys consumed or produced by the research aggregator."""

    DISTRIBUTION_DESCRIPTOR = "adaptive_distribution_descriptor"
    CLIENT_METRIC = "adaptive_client_metric"
    QUALITY_IMPROVEMENT = "adaptive_quality_improvement"
    FINAL_WEIGHTS = "adaptive_final_weights"
    MEAN_HETEROGENEITY = "adaptive_mean_heterogeneity"
    BLEND_FACTOR = "adaptive_blend_factor"


@dataclass
class _Contribution:
    name: str
    round_number: int
    data: dict
    sample_count: float
    descriptor: list[float]
    metric: float
    quality_improvement: float


class AdaptiveHeterogeneityAggregator(Aggregator):
    """Aggregate ``WEIGHT_DIFF`` updates using adaptive client weights.

    The class is designed for PyTorch ``FedOptRecipe``. FedOpt remains
    responsible for the server optimizer; this component changes only the
    weighted mean of client weight differences.

    Required client metadata:

    - ``MetaKey.NUM_STEPS_CURRENT_ROUND``: positive local-volume proxy;
    - ``AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR``: non-empty non-negative vector;
    - ``AdaptiveMetaKey.CLIENT_METRIC``: finite higher-is-better metric.

    ``AdaptiveMetaKey.QUALITY_IMPROVEMENT`` is optional and defaults to zero.
    """

    expected_data_kind = DataKind.WEIGHT_DIFF

    def __init__(
        self,
        sample_exponent: float = 0.65,
        representation_exponent: float = 0.70,
        quality_exponent: float = 0.40,
        fairness_strength: float = 1.00,
        heterogeneity_threshold: float = 0.26,
        heterogeneity_temperature: float = 0.04,
        heterogeneity_deadband: float = 0.15,
        max_blend_factor: float = 0.40,
        min_weight: float = 0.02,
        max_weight: float = 0.50,
    ):
        super().__init__()
        self.config = AdaptiveWeightingConfig(
            sample_exponent=sample_exponent,
            representation_exponent=representation_exponent,
            quality_exponent=quality_exponent,
            fairness_strength=fairness_strength,
            heterogeneity_threshold=heterogeneity_threshold,
            heterogeneity_temperature=heterogeneity_temperature,
            heterogeneity_deadband=heterogeneity_deadband,
            max_blend_factor=max_blend_factor,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        self.policy = AdaptiveHeterogeneityPolicy(self.config)
        self._contributions: dict[str, _Contribution] = {}
        self._processed_algorithm = None
        self._descriptor_size = None

    def reset(self, fl_ctx: FLContext):
        """Reset per-round state while retaining the configured policy."""
        self._contributions = {}
        self._processed_algorithm = None
        self._descriptor_size = None

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Validate and retain one client contribution for the current round."""
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            self.log_error(fl_ctx, f"expected {DataKind.WEIGHT_DIFF} but got {dxo.data_kind}")
            return False
        if not dxo.data:
            self.log_error(fl_ctx, "received empty weight-difference payload")
            return False

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding contribution from {contributor_name!r} at round {contribution_round}; "
                f"current round is {current_round}",
            )
            return False
        if contributor_name in self._contributions:
            self.log_warning(fl_ctx, f"discarding duplicate contribution from {contributor_name!r}")
            return False

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(fl_ctx, f"contributor {contributor_name!r} returned rc={rc}; contribution ignored")
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            if self._processed_algorithm is None:
                self._processed_algorithm = processed_algorithm
            elif self._processed_algorithm != processed_algorithm:
                self.log_error(
                    fl_ctx,
                    "all updates must use the same processed algorithm: "
                    f"expected {self._processed_algorithm!r}, got {processed_algorithm!r}",
                )
                return False

        sample_count = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
        descriptor = dxo.get_meta_prop(AdaptiveMetaKey.DISTRIBUTION_DESCRIPTOR)
        metric = dxo.get_meta_prop(AdaptiveMetaKey.CLIENT_METRIC)
        quality = dxo.get_meta_prop(AdaptiveMetaKey.QUALITY_IMPROVEMENT, 0.0)
        try:
            sample_count = float(sample_count)
            metric = float(metric)
            quality = float(quality)
            descriptor = [float(value) for value in descriptor]
        except (TypeError, ValueError):
            self.log_error(
                fl_ctx,
                f"contributor {contributor_name!r} is missing valid adaptive aggregation metadata",
            )
            return False

        if (
            sample_count <= 0.0
            or not math.isfinite(sample_count)
            or not math.isfinite(metric)
            or not math.isfinite(quality)
            or not descriptor
            or any(not math.isfinite(value) or value < 0.0 for value in descriptor)
            or sum(descriptor) <= 0.0
        ):
            self.log_error(fl_ctx, f"invalid adaptive aggregation metadata from {contributor_name!r}")
            return False

        if self._descriptor_size is None:
            self._descriptor_size = len(descriptor)
        elif len(descriptor) != self._descriptor_size:
            self.log_error(
                fl_ctx,
                f"descriptor size mismatch from {contributor_name!r}: "
                f"expected {self._descriptor_size}, got {len(descriptor)}",
            )
            return False

        self._contributions[contributor_name] = _Contribution(
            name=contributor_name,
            round_number=int(contribution_round),
            data=dxo.data,
            sample_count=sample_count,
            descriptor=descriptor,
            metric=metric,
            quality_improvement=quality,
        )
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Compute adaptive weights and return the weighted mean update."""
        if not self._contributions:
            raise ValueError("AdaptiveHeterogeneityAggregator cannot aggregate an empty contribution set")

        names = sorted(self._contributions)
        contributions = [self._contributions[name] for name in names]
        result = self.policy.compute(
            sample_counts=[item.sample_count for item in contributions],
            descriptors=[item.descriptor for item in contributions],
            client_metrics=[item.metric for item in contributions],
            quality_improvements=[item.quality_improvement for item in contributions],
        )

        helper = WeightedAggregationHelper()
        for item, weight in zip(contributions, result.weights):
            helper.add(
                data=item.data,
                weight=float(weight),
                contributor_name=item.name,
                contribution_round=item.round_number,
            )
        aggregated = helper.get_result()
        dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=aggregated,
            meta={
                AdaptiveMetaKey.FINAL_WEIGHTS: {name: float(weight) for name, weight in zip(names, result.weights)},
                AdaptiveMetaKey.MEAN_HETEROGENEITY: result.mean_heterogeneity,
                AdaptiveMetaKey.BLEND_FACTOR: result.blend_factor,
            },
        )
        if self._processed_algorithm is not None:
            dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, self._processed_algorithm)
        self.reset(fl_ctx)
        return dxo.to_shareable()
