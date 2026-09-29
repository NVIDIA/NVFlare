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
"""Schema-validating, sample-weighted aggregation for federated Evo2 updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import evo2_adapter_checkpoint as adapter_checkpoint
import provenance

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    WeightedAggregationHelper,
    filter_aggregatable_metrics,
)


class ExactSchemaFedAvgAggregator(ModelAggregator):
    """Validate every client DIFF against the initial checkpoint before FedAvg."""

    def __init__(
        self,
        schema_checkpoint: str,
        aggregation_weights: Mapping[str, float],
    ):
        super().__init__()
        self.schema_checkpoint = str(Path(schema_checkpoint).resolve())
        if not isinstance(aggregation_weights, Mapping) or not aggregation_weights:
            raise ValueError("aggregation_weights must be a non-empty mapping of client names to sample counts.")

        self.aggregation_weights = {}
        for client_name, weight in aggregation_weights.items():
            if not isinstance(client_name, str) or not client_name:
                raise ValueError(f"Aggregation weight client names must be non-empty strings, got {client_name!r}.")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or not math.isfinite(float(weight))
                or weight <= 0
            ):
                raise ValueError(f"Aggregation weight for {client_name!r} must be positive and finite, got {weight!r}.")
            self.aggregation_weights[client_name] = float(weight)

        self.reference_state = adapter_checkpoint.load_nvflare_checkpoint(self.schema_checkpoint)
        checkpoint_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(self.schema_checkpoint)
        self.initialization_metadata = provenance.resolve_initialization_metadata(checkpoint_metadata)
        if self.initialization_metadata.get("exchange_dtype") != adapter_checkpoint.EXCHANGE_DTYPE_NAME:
            raise ValueError(
                "Evo2 schema checkpoint initialization metadata must declare "
                f"exchange_dtype={adapter_checkpoint.EXCHANGE_DTYPE_NAME!r}."
            )
        self.reset_stats()

    def accept_model(self, model: FLModel):
        if model.params_type != ParamsType.DIFF:
            raise ValueError(f"Evo2 aggregation expects ParamsType.DIFF, received {model.params_type!r}.")

        metadata = model.meta or {}
        if metadata.get("exchange_dtype") != adapter_checkpoint.EXCHANGE_DTYPE_NAME:
            raise ValueError(
                "Evo2 client update metadata must declare "
                f"exchange_dtype={adapter_checkpoint.EXCHANGE_DTYPE_NAME!r}."
            )
        client_name = metadata.get("client_name")
        if client_name not in self.aggregation_weights:
            raise ValueError(f"Received Evo2 update from unconfigured client {client_name!r}.")
        if client_name in self.contributors:
            raise ValueError(f"Received more than one Evo2 update from {client_name!r} in the same round.")

        adapter_checkpoint.validate_trainable_state(
            model.params,
            self.reference_state,
            context=f"Evo2 DIFF from {client_name}",
        )
        self.param_contributions[client_name] = adapter_checkpoint.copy_trainable_state(
            model.params,
            context=f"Evo2 DIFF from {client_name}",
        )
        self.contributors.add(client_name)

        if model.metrics is None:
            self.all_metrics = False
        elif self.all_metrics:
            metrics = filter_aggregatable_metrics(model.metrics)
            if metrics:
                self.metric_contributions[client_name] = metrics

    def aggregate_model(self) -> FLModel:
        expected = set(self.aggregation_weights)
        if self.contributors != expected:
            missing = sorted(expected - self.contributors)
            unexpected = sorted(self.contributors - expected)
            raise RuntimeError(
                f"Cannot aggregate Evo2 round with incomplete contributors; missing={missing}, unexpected={unexpected}."
            )

        # Client tasks may finish in any order. Accumulate their canonical FP32
        # updates in client-name order so an otherwise identical round has stable
        # floating-point behavior independent of arrival timing.
        for client_name in sorted(self.param_contributions):
            self.params_helper.add(
                data=self.param_contributions[client_name],
                weight=self.aggregation_weights[client_name],
                contributor_name=client_name,
                contribution_round=None,
            )
        params = self.params_helper.get_result()
        adapter_checkpoint.validate_trainable_state(params, self.reference_state, context="Aggregated Evo2 DIFF")
        if self.all_metrics:
            for client_name in sorted(self.metric_contributions):
                self.metrics_helper.add(
                    data=self.metric_contributions[client_name],
                    weight=self.aggregation_weights[client_name],
                    contributor_name=client_name,
                    contribution_round=None,
                )
            metrics = self.metrics_helper.get_result()
        else:
            metrics = None
        return FLModel(
            params=params,
            params_type=ParamsType.DIFF,
            metrics=metrics or None,
            meta={
                "exchange_dtype": adapter_checkpoint.EXCHANGE_DTYPE_NAME,
                "initialization": self.initialization_metadata,
            },
        )

    def reset_stats(self):
        self.params_helper = WeightedAggregationHelper()
        self.metrics_helper = WeightedAggregationHelper()
        self.param_contributions = {}
        self.metric_contributions = {}
        self.contributors = set()
        self.all_metrics = True
