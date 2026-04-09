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
"""
Custom MedGemma aggregators for federated LoRA fine-tuning.

Implements fixed-global-rank naive and HLoRA-style server aggregators, based on:
"HLoRA: Towards Efficient Federated Fine-Tuning of Large Language Models with Heterogeneous LoRA".
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from lora_utils import get_lora_factor_pairs_and_base, is_lora_factor_key

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

WEIGHT_META_KEY = "num_examples"


def _factorize_hlora_update(
    weighted_b_factors: list[torch.Tensor],
    weighted_a_factors: list[torch.Tensor],
    target_rank: int,
    b_reference: torch.Tensor,
    a_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate weighted BA updates and project them back to a configured global LoRA rank.

    Instead of materializing a full dense average update and running an expensive full SVD,
    exploit the fact that the weighted sum of LoRA updates remains low-rank:

        sum_k w_k B_k A_k = [sqrt(w_k) B_k]_k [sqrt(w_k) A_k]_k

    A compact QR + SVD on the reduced factors yields the same truncated rank-r projection
    with much lower server memory use.
    """

    left = torch.cat(weighted_b_factors, dim=1)
    right = torch.cat(weighted_a_factors, dim=0)

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.transpose(0, 1), mode="reduced")
    middle = r_left @ r_right.transpose(0, 1)

    aggregated_b = torch.zeros((b_reference.shape[0], target_rank), dtype=torch.float32)
    aggregated_a = torch.zeros((target_rank, a_reference.shape[1]), dtype=torch.float32)
    if middle.numel() == 0:
        return (
            aggregated_b.to(dtype=b_reference.dtype).contiguous(),
            aggregated_a.to(dtype=a_reference.dtype).contiguous(),
        )

    u_hat, singular_values, vh_hat = torch.linalg.svd(middle, full_matrices=False)
    effective_rank = min(target_rank, singular_values.shape[0])
    if effective_rank == 0:
        return (
            aggregated_b.to(dtype=b_reference.dtype).contiguous(),
            aggregated_a.to(dtype=a_reference.dtype).contiguous(),
        )

    u = q_left @ u_hat[:, :effective_rank]
    vh = vh_hat[:effective_rank, :] @ q_right.transpose(0, 1)
    aggregated_a[:effective_rank, :] = torch.diag(singular_values[:effective_rank]) @ vh
    aggregated_b[:, :effective_rank] = u
    return (
        aggregated_b.to(dtype=b_reference.dtype).contiguous(),
        aggregated_a.to(dtype=a_reference.dtype).contiguous(),
    )


def _factorize_naive_average(
    client_lora_params: list[tuple[float, dict[str, torch.Tensor]]],
    total_weight: float,
    a_key: str,
    b_key: str,
    target_rank: int,
    b_reference: torch.Tensor,
    a_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aggregated_b = torch.zeros((b_reference.shape[0], target_rank), dtype=torch.float32)
    aggregated_a = torch.zeros((target_rank, a_reference.shape[1]), dtype=torch.float32)
    for weight, client_state in client_lora_params:
        weight_fraction = weight / total_weight
        b_factor = client_state[b_key]
        a_factor = client_state[a_key]
        active_rank = min(target_rank, b_factor.shape[1], a_factor.shape[0])
        if active_rank == 0:
            continue
        aggregated_b[:, :active_rank] += b_factor[:, :active_rank] * weight_fraction
        aggregated_a[:active_rank, :] += a_factor[:active_rank, :] * weight_fraction

    return (
        aggregated_b.to(dtype=b_reference.dtype).contiguous(),
        aggregated_a.to(dtype=a_reference.dtype).contiguous(),
    )


class _BaseMaxRankLoraAggregator(ModelAggregator):
    def __init__(
        self,
        *,
        global_lora_rank: int,
        global_rank_map: dict[str, int] | None = None,
        weight_meta_key: str = WEIGHT_META_KEY,
    ):
        super().__init__()
        if global_lora_rank <= 0:
            raise ValueError(f"global_lora_rank must be positive, got {global_lora_rank}.")
        self.global_lora_rank = global_lora_rank
        self.global_rank_map = dict(global_rank_map or {})
        self.weight_meta_key = weight_meta_key
        self.total_weight = 0.0
        self.params_type = None
        self.expected_param_keys = None
        self.non_lora_weighted_sum = {}
        self.param_references = {}
        self.client_lora_params = []
        self.metric_weighted_sum = {}
        self.metric_total_weight = {}
        self.all_metrics = True

    def _get_weight(self, model: FLModel) -> float:
        meta = model.meta or {}
        weight = meta.get(self.weight_meta_key, meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        if weight is None:
            return 1.0
        return float(weight)

    def _resolve_global_rank_map(self, lora_keys: list[str]) -> dict[str, int]:
        resolved = dict(self.global_rank_map)
        for _a_key, _b_key, base_key in get_lora_factor_pairs_and_base(lora_keys):
            resolved.setdefault(base_key, self.global_lora_rank)
        for base_key, rank in resolved.items():
            if rank <= 0:
                raise ValueError(f"Configured global LoRA rank must be positive for {base_key}, got {rank}.")
        return resolved

    def accept_model(self, model: FLModel):
        if model.params_type != ParamsType.FULL:
            raise ValueError("Max-rank LoRA aggregation expects FULL LoRA factor weights, not DIFF updates.")

        weight = self._get_weight(model)
        if weight <= 0:
            raise ValueError(f"Aggregation weight must be positive, got {weight}.")

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}. "
                "All client models must have the same params_type."
            )

        param_keys = set(model.params.keys())
        if self.expected_param_keys is None:
            self.expected_param_keys = param_keys
        elif param_keys != self.expected_param_keys:
            raise ValueError("All client models must have the same parameter keys for max-rank LoRA aggregation.")

        client_lora_state = {}
        for key, value in model.params.items():
            tensor = value.detach().cpu().contiguous()
            self.param_references.setdefault(key, tensor)
            if is_lora_factor_key(key):
                client_lora_state[key] = tensor.to(dtype=torch.float32)
            else:
                if key not in self.non_lora_weighted_sum:
                    self.non_lora_weighted_sum[key] = tensor.to(dtype=torch.float32) * weight
                else:
                    self.non_lora_weighted_sum[key] += tensor.to(dtype=torch.float32) * weight

        self.client_lora_params.append((weight, client_lora_state))
        self.total_weight += weight
        self._accept_metrics(model.metrics, weight)

    def _accept_metrics(self, metrics, weight: float) -> None:
        if metrics is None:
            self.all_metrics = False
            return

        for key, value in metrics.items():
            if not isinstance(value, Real):
                continue
            self.metric_weighted_sum[key] = self.metric_weighted_sum.get(key, 0.0) + float(value) * weight
            self.metric_total_weight[key] = self.metric_total_weight.get(key, 0.0) + weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate!")
            return FLModel(params_type=ParamsType.FULL, params={})

        aggregated_params = {}
        for key, value in self.non_lora_weighted_sum.items():
            reference = self.param_references[key]
            aggregated_params[key] = (value / self.total_weight).to(dtype=reference.dtype).contiguous()

        if self.client_lora_params:
            lora_keys = list(self.client_lora_params[0][1].keys())
            lora_pairs = get_lora_factor_pairs_and_base(lora_keys)
            resolved_rank_map = self._resolve_global_rank_map(lora_keys)
            self.info(
                f"{self._aggregation_name()} aggregation: "
                f"clients={len(self.client_lora_params)} "
                f"lora_pairs={len(lora_pairs)} "
                f"aux_tensors={len(self.non_lora_weighted_sum)} "
                f"global_rank={self.global_lora_rank} "
                f"weight_meta_key={self.weight_meta_key}"
            )
            for a_key, b_key, base_key in lora_pairs:
                target_rank = resolved_rank_map[base_key]
                aggregated_b, aggregated_a = self._aggregate_lora_pair(
                    a_key=a_key,
                    b_key=b_key,
                    target_rank=target_rank,
                )
                aggregated_params[a_key] = aggregated_a
                aggregated_params[b_key] = aggregated_b

        aggregated_metrics = None
        if self.all_metrics:
            aggregated_metrics = {
                key: self.metric_weighted_sum[key] / self.metric_total_weight[key]
                for key in self.metric_weighted_sum
                if self.metric_total_weight.get(key)
            }

        return FLModel(params=aggregated_params, params_type=self.params_type, metrics=aggregated_metrics)

    def _aggregation_name(self) -> str:
        raise NotImplementedError()

    def _aggregate_lora_pair(self, a_key: str, b_key: str, target_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def reset_stats(self):
        self.total_weight = 0.0
        self.params_type = None
        self.expected_param_keys = None
        self.non_lora_weighted_sum = {}
        self.param_references = {}
        self.client_lora_params = []
        self.metric_weighted_sum = {}
        self.metric_total_weight = {}
        self.all_metrics = True


class NaiveMaxRankAggregator(_BaseMaxRankLoraAggregator):
    """Weighted factor averaging with a fixed server-side max-rank LoRA bank."""

    def _aggregation_name(self) -> str:
        return "Naive max-rank LoRA"

    def _aggregate_lora_pair(self, a_key: str, b_key: str, target_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _factorize_naive_average(
            client_lora_params=self.client_lora_params,
            total_weight=self.total_weight,
            a_key=a_key,
            b_key=b_key,
            target_rank=target_rank,
            b_reference=self.param_references[b_key],
            a_reference=self.param_references[a_key],
        )


class HLoRAMaxRankAggregator(_BaseMaxRankLoraAggregator):
    """HLoRA-style aggregation with a fixed server-side max-rank LoRA bank."""

    def _aggregation_name(self) -> str:
        return "HLoRA max-rank LoRA"

    def _aggregate_lora_pair(self, a_key: str, b_key: str, target_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_b_factors = []
        weighted_a_factors = []
        for weight, client_state in self.client_lora_params:
            sqrt_weight = math.sqrt(weight / self.total_weight)
            weighted_b_factors.append(client_state[b_key] * sqrt_weight)
            weighted_a_factors.append(client_state[a_key] * sqrt_weight)

        return _factorize_hlora_update(
            weighted_b_factors=weighted_b_factors,
            weighted_a_factors=weighted_a_factors,
            target_rank=target_rank,
            b_reference=self.param_references[b_key],
            a_reference=self.param_references[a_key],
        )
