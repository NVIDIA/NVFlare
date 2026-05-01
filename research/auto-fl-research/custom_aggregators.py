# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Custom aggregator implementations for the Auto-FL NVFlare starter.

These are NVFlare-oriented aggregation variants intended for bounded autoresearch experiments.
The repo-level loop is inspired by the public karpathy/autoresearch operating model, but the
aggregation code itself is adapted to NVFlare's FLModel / ModelAggregator interfaces.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:
    torch = None

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

try:
    from nvflare.app_common.app_constant import AlgorithmConstants

    SCAFFOLD_CTRL_DIFF = AlgorithmConstants.SCAFFOLD_CTRL_DIFF
    SCAFFOLD_CTRL_GLOBAL = AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL
except Exception:
    SCAFFOLD_CTRL_DIFF = "scaffold_c_diff"
    SCAFFOLD_CTRL_GLOBAL = "scaffold_c_global"


def _as_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_output_type(value, reference):
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if isinstance(reference, np.ndarray):
        return np.asarray(value, dtype=reference.dtype)
    return np.asarray(value)


def _to_meta_numpy(value, reference):
    result = np.asarray(value)
    if reference is None:
        return result
    ref_array = _as_numpy(reference)
    return result.astype(ref_array.dtype, copy=False)


class WeightedAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        weight = model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate")
            return FLModel(params={})

        aggregated_params = {key: val / self.total_weight for key, val in self.weighted_sum.items()}
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None


class FedAvgAggregator(WeightedAggregator):
    """Explicit FedAvg alias for benchmark readability."""


class FedOptAggregator(ModelAggregator):
    """Server-side optimizer over weighted client DIFFs.

    This keeps the FL contract intact: clients still send model DIFFs with
    NUM_STEPS_CURRENT_ROUND, and the server returns a DIFF update with the same
    parameter keys and params_type.
    """

    def __init__(
        self,
        optimizer: str = "sgdm",
        server_lr: float = 1.0,
        server_momentum: float = 0.6,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ):
        super().__init__()
        if optimizer not in {"sgdm", "adam"}:
            raise ValueError(f"Unsupported FedOpt optimizer: {optimizer}")
        if server_lr <= 0.0:
            raise ValueError("server_lr must be > 0")
        if not 0.0 <= server_momentum < 1.0:
            raise ValueError("server_momentum must be in [0, 1)")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if tau <= 0.0:
            raise ValueError("tau must be > 0")

        self.optimizer = optimizer
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        self.first_moment = {}
        self.second_moment = {}
        self.adam_step = 0
        self.reset_stats()

    def accept_model(self, model: FLModel):
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.references.setdefault(key, value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = diff * weight
            else:
                self.weighted_sum[key] += diff * weight
        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate")
            return FLModel(params={})

        mean_diff = {key: val / self.total_weight for key, val in self.weighted_sum.items()}
        if self.optimizer == "sgdm":
            update = self._sgdm_update(mean_diff)
        else:
            update = self._adam_update(mean_diff)

        aggregated_params = {key: _to_output_type(update[key], self.references[key]) for key in update}
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        self.weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None
        self.references = {}

    def _sgdm_update(self, mean_diff):
        updates = {}
        for key, diff in mean_diff.items():
            previous = self.first_moment.get(key)
            if previous is None:
                previous = np.zeros_like(diff)
            velocity = self.server_momentum * previous + diff
            self.first_moment[key] = velocity
            updates[key] = self.server_lr * velocity
        return updates

    def _adam_update(self, mean_diff):
        updates = {}
        self.adam_step += 1
        first_bias_correction = 1.0 - self.beta1**self.adam_step
        second_bias_correction = 1.0 - self.beta2**self.adam_step

        for key, diff in mean_diff.items():
            first = self.first_moment.get(key)
            if first is None:
                first = np.zeros_like(diff)
            second = self.second_moment.get(key)
            if second is None:
                second = np.zeros_like(diff)

            first = self.beta1 * first + (1.0 - self.beta1) * diff
            second = self.beta2 * second + (1.0 - self.beta2) * np.square(diff)
            self.first_moment[key] = first
            self.second_moment[key] = second
            first_hat = first / first_bias_correction
            second_hat = second / second_bias_correction
            updates[key] = self.server_lr * first_hat / (np.sqrt(second_hat) + self.tau)
        return updates


class FedAvgMAggregator(FedOptAggregator):
    def __init__(self, server_lr: float = 1.0, server_momentum: float = 0.6):
        super().__init__(
            optimizer="sgdm",
            server_lr=server_lr,
            server_momentum=server_momentum,
        )


class FedAdamAggregator(FedOptAggregator):
    def __init__(
        self,
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ):
        super().__init__(
            optimizer="adam",
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
        )


class ScaffoldAggregator(ModelAggregator):
    """SCAFFOLD aggregation over DIFF params plus control-variate metadata."""

    def __init__(self):
        super().__init__()
        self.global_controls = {}
        self.reset_stats()

    def accept_model(self, model: FLModel):
        weight = float(model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        self.client_weights.append(weight)

        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")

        for key, value in model.params.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.references.setdefault(key, value)
            if key not in self.weighted_sum:
                self.weighted_sum[key] = diff * weight
            else:
                self.weighted_sum[key] += diff * weight

        ctrl_diff = model.meta.get(SCAFFOLD_CTRL_DIFF)
        if not ctrl_diff:
            client_name = model.meta.get("site_name", "unknown")
            raise ValueError(
                f"Client '{client_name}' did not return required "
                f"FLModel.meta['{SCAFFOLD_CTRL_DIFF}'] for SCAFFOLD aggregation."
            )
        for key, value in ctrl_diff.items():
            diff = _as_numpy(value).astype(np.float64, copy=False)
            self.control_references.setdefault(key, value)
            if key not in self.control_weighted_sum:
                self.control_weighted_sum[key] = diff * weight
            else:
                self.control_weighted_sum[key] += diff * weight

        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate")
            return FLModel(params={})

        aggregated_params = {
            key: _to_output_type(val / self.total_weight, self.references[key])
            for key, val in self.weighted_sum.items()
        }

        for key, val in self.control_weighted_sum.items():
            delta = val / self.total_weight
            previous = self.global_controls.get(key)
            if previous is None:
                previous = np.zeros_like(delta)
            self.global_controls[key] = previous + delta

        global_controls = {
            key: _to_meta_numpy(value, self.control_references.get(key)) for key, value in self.global_controls.items()
        }
        return FLModel(
            params=aggregated_params,
            params_type=self.params_type,
            meta={SCAFFOLD_CTRL_GLOBAL: global_controls},
        )

    def reset_stats(self):
        self.weighted_sum = {}
        self.control_weighted_sum = {}
        self.total_weight = 0.0
        self.client_weights = []
        self.params_type = None
        self.references = {}
        self.control_references = {}


class MedianAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.client_models = []
        self.params_type = None

    def accept_model(self, model: FLModel):
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}.")
        self.client_models.append(model.params)

    def aggregate_model(self) -> FLModel:
        if not self.client_models:
            self.error("No client models to aggregate")
            return FLModel(params={})

        aggregated_params = {}
        param_keys = self.client_models[0].keys()
        for key in param_keys:
            stacked = np.stack([_as_numpy(m[key]) for m in self.client_models], axis=0)
            aggregated_params[key] = np.median(stacked, axis=0)

        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        self.client_models = []
        self.params_type = None
