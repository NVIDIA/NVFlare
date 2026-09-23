# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""FedAvg aggregation with the same update-norm clipping used by FedSCS."""

from typing import Dict, Optional

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper


class FedAvgClippedAggregator(ModelAggregator):
    """Standard NVFLARE FedAvg with client-update L2 clipping."""

    # Bound client-reported step weights so a malicious metadata value
    # cannot create an arbitrarily large aggregation contribution.
    MAX_NUM_STEPS_WEIGHT = 1_000_000.0

    def __init__(
        self,
        max_update_norm: Optional[float] = 10.0,
        aggregation_weights: Optional[Dict[str, float]] = None,
        expected_schema: Optional[Dict[str, tuple]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        if max_update_norm is not None:
            if not np.isfinite(max_update_norm) or max_update_norm <= 0:
                raise ValueError("max_update_norm must be positive and finite")

        self.max_update_norm = max_update_norm
        self.aggregation_weights = aggregation_weights or {}
        self.expected_schema = expected_schema
        self.expected_dtypes = expected_dtypes

        if (expected_schema is None) != (expected_dtypes is None):
            raise ValueError("expected_schema and expected_dtypes must be provided together")

        if expected_schema is not None:
            if set(expected_schema) != set(expected_dtypes):
                raise ValueError("expected_schema and expected_dtypes must contain the same keys")

        self._aggr_helper = WeightedAggregationHelper()
        self._aggr_metrics_helper = WeightedAggregationHelper()

        self._params_type = None
        self._current_round = 0

    def _get_client_name(self, model: FLModel) -> str:
        client_name = model.meta.get("client_name")

        if client_name is None:
            raise ValueError("client_name is required in model metadata")

        return str(client_name)

    def _validate_schema(self, params: Dict[str, np.ndarray], client_name: str):
        if self.expected_schema is None:
            return

        expected_keys = set(self.expected_schema)
        received_keys = set(params)

        missing = sorted(expected_keys - received_keys)
        extra = sorted(received_keys - expected_keys)

        if missing or extra:
            raise ValueError(
                f"Parameter schema mismatch for client {client_name}: " f"missing={missing}, extra={extra}"
            )

        for name, value in params.items():
            expected_shape = tuple(self.expected_schema[name])
            received_shape = tuple(np.asarray(value).shape)

            if received_shape != expected_shape:
                raise ValueError(
                    f"Parameter shape mismatch for client {client_name}, "
                    f"{name}: expected {expected_shape}, got {received_shape}"
                )

            if self.expected_dtypes is not None:
                expected_dtype = self.expected_dtypes[name]
                received_dtype = str(np.asarray(value).dtype)

                if received_dtype != expected_dtype:
                    raise ValueError(
                        f"Parameter dtype mismatch for client {client_name}, "
                        f"{name}: expected {expected_dtype}, got {received_dtype}"
                    )

    def _get_num_steps_weight(self, model: FLModel) -> float:
        value = model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND)

        if value is None or isinstance(value, bool):
            return 1.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 1.0

        if not np.isfinite(value) or value <= 0:
            return 1.0

        return min(value, self.MAX_NUM_STEPS_WEIGHT)

    def _get_aggregation_weight(self, client_name: str) -> float:
        value = self.aggregation_weights.get(client_name, 1.0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid aggregation weight for client {client_name}") from None

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Aggregation weight for client {client_name} must be positive and finite")

        return value

    def _clip_params(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.max_update_norm is None:
            return params

        flat_parts = []

        for value in params.values():
            array = np.asarray(value)

            if not np.all(np.isfinite(array)):
                raise ValueError("Client update contains non-finite values")

            flat_parts.append(array.astype(np.float64, copy=False).reshape(-1))

        if not flat_parts:
            return params

        flat = np.concatenate(flat_parts)
        norm = float(np.linalg.norm(flat))

        if not np.isfinite(norm):
            raise ValueError("Client update norm is non-finite")

        if norm <= self.max_update_norm:
            return params

        scale = self.max_update_norm / norm

        clipped = {}
        for name, value in params.items():
            array = np.asarray(value)
            clipped[name] = (array.astype(np.float64) * scale).astype(array.dtype, copy=False)

        return clipped

    def accept_model(self, model: FLModel) -> bool:
        if model.params is None:
            raise ValueError("Model parameters are required")

        if model.params_type != "DIFF":
            raise ValueError(f"FedAvgClippedAggregator expects DIFF, got {model.params_type}")

        client_name = self._get_client_name(model)
        self._validate_schema(model.params, client_name)

        params = self._clip_params(model.params)

        if self._params_type is None:
            self._params_type = model.params_type

        if model.current_round is not None:
            self._current_round = model.current_round

        aggregation_weight = self._get_aggregation_weight(client_name)
        num_steps_weight = self._get_num_steps_weight(model)
        weight = aggregation_weight * num_steps_weight

        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid effective aggregation weight for client {client_name}: {weight}")

        self._aggr_helper.add(
            data=params,
            weight=weight,
            contributor_name=client_name,
            contribution_round=model.current_round,
        )

        if model.metrics:
            self._aggr_metrics_helper.add(
                data=model.metrics,
                weight=weight,
                contributor_name=client_name,
                contribution_round=model.current_round,
            )

        return True

    def aggregate_model(self) -> FLModel:
        if self._params_type is None:
            raise RuntimeError("No client models have been accepted")

        aggregated_params = self._aggr_helper.get_result()
        aggregated_metrics = self._aggr_metrics_helper.get_result()

        result = FLModel(
            params=aggregated_params,
            params_type=self._params_type,
            current_round=self._current_round,
            metrics=aggregated_metrics,
            meta={},
        )

        self.reset_stats()

        return result

    def reset_stats(self):
        self._aggr_helper = WeightedAggregationHelper()
        self._aggr_metrics_helper = WeightedAggregationHelper()
        self._params_type = None
        self._current_round = 0
