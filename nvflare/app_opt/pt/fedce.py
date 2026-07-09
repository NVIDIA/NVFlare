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

"""PyTorch components for FedCE contribution-aware aggregation."""

import copy
import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    WeightedAggregationHelper,
    filter_aggregatable_metrics,
)


class FedCEConstants:
    """Metadata keys shared by the FedCE server and client training script."""

    MINUS_MODEL_SCORE = "fedce_minus_val"
    CONTRIBUTION_WEIGHTS = "fedce_coef"


class PTFedCEHelper:
    """Client-side utilities for the FedCE-specific training contract."""

    @staticmethod
    def get_contribution_weight(global_model: FLModel, client_name: str, default: float = 0.0) -> float:
        weights = (global_model.meta or {}).get(FedCEConstants.CONTRIBUTION_WEIGHTS, {})
        return float(weights.get(client_name, default))

    @staticmethod
    def make_minus_model(global_model, previous_local_state: Dict, contribution_weight: float):
        """Construct the FedCE leave-one-out model for local validation.

        ``previous_local_state`` is the client's trained model state from the
        preceding round. Non-floating buffers are copied from the global model.
        """
        if not 0.0 <= contribution_weight < 1.0:
            raise ValueError(f"contribution_weight must be in [0, 1), got {contribution_weight}")
        minus_model = copy.deepcopy(global_model)
        denominator = 1.0 - contribution_weight
        global_state = global_model.state_dict()
        minus_state = minus_model.state_dict()
        for name, global_value in global_state.items():
            if name not in previous_local_state:
                continue
            if not (global_value.is_floating_point() or global_value.is_complex()):
                minus_state[name].copy_(global_value)
                continue
            previous_value = torch.as_tensor(
                previous_local_state[name],
                device=global_value.device,
                dtype=global_value.dtype,
            )
            minus_state[name].copy_((global_value - contribution_weight * previous_value) / denominator)
        return minus_model

    @staticmethod
    def set_minus_model_score(result: FLModel, score: float) -> FLModel:
        """Attach a contribution score, where larger values mean greater contribution."""
        result.meta[FedCEConstants.MINUS_MODEL_SCORE] = float(score)
        return result


def _normalize(values: List[float], epsilon: float) -> List[float]:
    if not values:
        raise ValueError("cannot normalize an empty value list")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"FedCE contribution values must be finite, got {values}")
    clipped = [max(float(value), epsilon) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in clipped]


class FedCEModelAggregator(ModelAggregator):
    """Aggregate client weight differences using FedCE contribution estimates.

    FedCE combines two signals: gradient-direction novelty and a client-provided
    score obtained by validating a leave-one-out ("minus") model. The resulting
    contribution weights are returned in the global model metadata so clients can
    construct the next round's minus model.

    Client results must use ``ParamsType.DIFF`` and include
    ``FLModel.meta[FedCEConstants.MINUS_MODEL_SCORE]``.
    """

    MODES = ("plus", "times")

    def __init__(
        self,
        mode: str = "plus",
        trainable_param_names: Optional[List[str]] = None,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be greater than 0, got {epsilon}")

        self.mode = mode
        self.trainable_param_names = list(trainable_param_names or [])
        self.epsilon = epsilon
        self._results: Dict[str, FLModel] = {}
        self._contribution_weights: Dict[str, float] = {}
        self._cosine_history: Dict[str, List[float]] = defaultdict(list)

    def reset_stats(self):
        self._results = {}

    def accept_model(self, model: FLModel):
        client_name = (model.meta or {}).get("client_name")
        if not client_name:
            raise ValueError("FedCE client result is missing FLModel.meta['client_name']")
        if model.params_type != ParamsType.DIFF:
            raise ValueError(
                f"FedCE requires ParamsType.DIFF results, got {model.params_type} from client {client_name!r}"
            )
        if not model.params:
            raise ValueError(f"FedCE received empty parameters from client {client_name!r}")
        if FedCEConstants.MINUS_MODEL_SCORE not in model.meta:
            raise ValueError(
                f"FedCE client {client_name!r} did not return required metadata "
                f"{FedCEConstants.MINUS_MODEL_SCORE!r}"
            )
        if client_name in self._results:
            raise ValueError(f"FedCE received more than one result from client {client_name!r} in the same round")
        task_meta = model.meta.get("props", {})
        prior_weights = task_meta.get(FedCEConstants.CONTRIBUTION_WEIGHTS) if isinstance(task_meta, dict) else None
        if not self._contribution_weights and isinstance(prior_weights, dict):
            self._contribution_weights = {name: float(weight) for name, weight in prior_weights.items()}
        self._results[client_name] = model

    def aggregate_model(self) -> FLModel:
        if not self._results:
            raise ValueError("FedCE cannot aggregate an empty result set")

        clients = sorted(self._results)
        param_names = self._get_cosine_param_names(clients)
        prior_weights = self._get_prior_weights(clients)
        consensus = self._weighted_vector(clients, prior_weights, param_names)

        cosine_scores = []
        minus_scores = []
        for client in clients:
            client_vector = self._flatten_params(self._results[client].params, param_names)
            without_client = consensus - prior_weights[client] * client_vector
            if torch.linalg.vector_norm(client_vector) == 0 or torch.linalg.vector_norm(without_client) == 0:
                similarity = 0.0
            else:
                similarity = float(torch.cosine_similarity(client_vector, without_client, dim=0).item())
            self._cosine_history[client].append(similarity)
            mean_similarity = sum(self._cosine_history[client]) / len(self._cosine_history[client])
            cosine_scores.append(max(0.0, 1.0 - mean_similarity))
            minus_scores.append(float(self._results[client].meta[FedCEConstants.MINUS_MODEL_SCORE]))

        cosine_weights = _normalize(cosine_scores, self.epsilon)
        minus_weights = _normalize(minus_scores, self.epsilon)
        if self.mode == "plus":
            combined = [cosine + minus for cosine, minus in zip(cosine_weights, minus_weights)]
        else:
            combined = [cosine * minus for cosine, minus in zip(cosine_weights, minus_weights)]
        normalized = _normalize(combined, self.epsilon)
        self._contribution_weights = dict(zip(clients, normalized))

        params_helper = WeightedAggregationHelper()
        metrics_helper = WeightedAggregationHelper()
        all_metrics = True
        current_round = None
        for client in clients:
            result = self._results[client]
            current_round = result.current_round
            weight = self._contribution_weights[client]
            params_helper.add(
                data=result.params,
                weight=weight,
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
                        weight=weight,
                        contributor_name=client,
                        contribution_round=current_round,
                    )

        metrics = metrics_helper.get_result() if all_metrics and metrics_helper.total else None
        return FLModel(
            params=params_helper.get_result(),
            params_type=ParamsType.DIFF,
            metrics=metrics or None,
            current_round=current_round,
            meta={
                FedCEConstants.CONTRIBUTION_WEIGHTS: dict(self._contribution_weights),
                "nr_aggregated": len(clients),
            },
        )

    def _get_cosine_param_names(self, clients: List[str]) -> List[str]:
        common = set(self._results[clients[0]].params)
        for client in clients[1:]:
            common.intersection_update(self._results[client].params)
        if self.trainable_param_names:
            common.intersection_update(self.trainable_param_names)
        if not common:
            raise ValueError("FedCE client updates have no common trainable parameters")
        return sorted(common)

    def _get_prior_weights(self, clients: List[str]) -> Dict[str, float]:
        values = [self._contribution_weights.get(client, 1.0) for client in clients]
        return dict(zip(clients, _normalize(values, self.epsilon)))

    def _weighted_vector(self, clients: List[str], weights: Dict[str, float], param_names: List[str]):
        result = None
        for client in clients:
            vector = self._flatten_params(self._results[client].params, param_names)
            weighted = weights[client] * vector
            result = weighted if result is None else result + weighted
        return result

    @staticmethod
    def _flatten_params(params: Dict, param_names: List[str]) -> torch.Tensor:
        tensors = []
        for name in param_names:
            value = params[name]
            materialize = getattr(value, "materialize", None)
            if callable(materialize):
                value = materialize()
            tensors.append(torch.as_tensor(value).detach().reshape(-1).to(device="cpu", dtype=torch.float32))
        return torch.cat(tensors)
