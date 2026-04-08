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

Implements an HLoRA-style server aggregator for homogeneous-rank LoRA, based on:
"HLoRA: Towards Efficient Federated Fine-Tuning of Large Language Models with Heterogeneous LoRA".
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


def _get_round_weight(model: FLModel) -> float:
    meta = model.meta or {}
    n_iter = meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
    if n_iter is None:
        return 1.0
    return float(n_iter)


def _is_lora_a_key(key: str) -> bool:
    return ".lora_A." in key


def _is_lora_b_key(key: str) -> bool:
    return ".lora_B." in key


def _is_lora_factor_key(key: str) -> bool:
    return _is_lora_a_key(key) or _is_lora_b_key(key)


def _get_lora_factor_pairs(keys: list[str]) -> list[tuple[str, str]]:
    pairs = {}
    for key in keys:
        if _is_lora_a_key(key):
            base = key.replace(".lora_A.", ".", 1)
            pairs.setdefault(base, {})["a"] = key
        elif _is_lora_b_key(key):
            base = key.replace(".lora_B.", ".", 1)
            pairs.setdefault(base, {})["b"] = key

    missing = [base for base, value in pairs.items() if "a" not in value or "b" not in value]
    if missing:
        raise ValueError(f"Missing LoRA A/B factor pairs for keys: {missing[:3]}")

    return [(pairs[base]["a"], pairs[base]["b"]) for base in sorted(pairs)]


def _factorize_hlora_update(
    weighted_b_factors: list[torch.Tensor],
    weighted_a_factors: list[torch.Tensor],
    b_template: torch.Tensor,
    a_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate weighted BA updates and project them back to the original LoRA rank.

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

    aggregated_b = torch.zeros_like(b_template)
    aggregated_a = torch.zeros_like(a_template)
    if middle.numel() == 0:
        return aggregated_b.contiguous(), aggregated_a.contiguous()

    u_hat, singular_values, vh_hat = torch.linalg.svd(middle, full_matrices=False)
    effective_rank = min(b_template.shape[1], singular_values.shape[0])
    if effective_rank == 0:
        return aggregated_b.contiguous(), aggregated_a.contiguous()

    u = q_left @ u_hat[:, :effective_rank]
    vh = vh_hat[:effective_rank, :] @ q_right.transpose(0, 1)
    aggregated_a[:effective_rank, :] = (torch.diag(singular_values[:effective_rank]) @ vh).to(dtype=aggregated_a.dtype)
    aggregated_b[:, :effective_rank] = u.to(dtype=aggregated_b.dtype)
    return aggregated_b.contiguous(), aggregated_a.contiguous()


class HLoraAggregator(ModelAggregator):
    """HLoRA-style aggregation for LoRA factors plus weighted averaging for other tensors."""

    def __init__(self):
        super().__init__()
        self.total_weight = 0.0
        self.params_type = None
        self.expected_param_keys = None
        self.non_lora_weighted_sum = {}
        self.client_lora_params = []
        self.metric_weighted_sum = {}
        self.metric_total_weight = {}
        self.all_metrics = True

    def accept_model(self, model: FLModel):
        weight = _get_round_weight(model)
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
            raise ValueError("All client models must have the same parameter keys for HLoRA aggregation.")

        client_lora_state = {}
        for key, value in model.params.items():
            if _is_lora_factor_key(key):
                client_lora_state[key] = value.detach().cpu().contiguous()
            else:
                if key not in self.non_lora_weighted_sum:
                    self.non_lora_weighted_sum[key] = value * weight
                else:
                    self.non_lora_weighted_sum[key] += value * weight

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
            return FLModel(params={})

        aggregated_params = {key: value / self.total_weight for key, value in self.non_lora_weighted_sum.items()}
        if self.client_lora_params:
            lora_keys = list(self.client_lora_params[0][1].keys())
            lora_pairs = _get_lora_factor_pairs(lora_keys)
            self.info(
                "HLoRA aggregation: "
                f"clients={len(self.client_lora_params)} "
                f"lora_pairs={len(lora_pairs)} "
                f"aux_tensors={len(self.non_lora_weighted_sum)}"
            )
            for a_key, b_key in lora_pairs:
                weighted_b_factors = []
                weighted_a_factors = []
                b_template = self.client_lora_params[0][1][b_key]
                a_template = self.client_lora_params[0][1][a_key]

                for weight, client_state in self.client_lora_params:
                    b_factor = client_state[b_key].to(dtype=torch.float32)
                    a_factor = client_state[a_key].to(dtype=torch.float32)
                    sqrt_weight = math.sqrt(weight / self.total_weight)
                    weighted_b_factors.append(b_factor * sqrt_weight)
                    weighted_a_factors.append(a_factor * sqrt_weight)

                aggregated_b, aggregated_a = _factorize_hlora_update(
                    weighted_b_factors=weighted_b_factors,
                    weighted_a_factors=weighted_a_factors,
                    b_template=b_template,
                    a_template=a_template,
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

    def reset_stats(self):
        self.total_weight = 0.0
        self.params_type = None
        self.expected_param_keys = None
        self.non_lora_weighted_sum = {}
        self.client_lora_params = []
        self.metric_weighted_sum = {}
        self.metric_total_weight = {}
        self.all_metrics = True
