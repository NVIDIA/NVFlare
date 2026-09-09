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

"""Versioning and canonical configuration for comparable CIFAR-10 runs."""

import hashlib
import json

PROTOCOL_VERSION = "cifar10_dirichlet_trainval_test_v3"


def canonical_config_hash(config: dict) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible configuration."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def common_run_config(
    n_clients: int,
    num_rounds: int,
    aggregation_epochs: int,
    batch_size: int,
    lr: float,
    validation_fraction: float,
    num_workers: int,
    num_threads: int | None,
    gpu_config: str | None,
    eval_batch_size: int,
    eval_num_workers: int,
    eval_device: str | None,
) -> dict:
    """Settings that must be shared by every method in a comparison campaign."""

    return {
        "dataset": "CIFAR-10",
        "model": "ModerateCNN",
        "partitioner": "NVFlare Dirichlet",
        "n_clients": int(n_clients),
        "num_rounds": int(num_rounds),
        "aggregation_epochs": int(aggregation_epochs),
        "batch_size": int(batch_size),
        "client_lr": float(lr),
        "client_optimizer": "torch.optim.SGD",
        "client_momentum": 0.9,
        "client_scheduler": "CosineAnnealingLR",
        "cosine_lr_eta_min_factor": 0.01,
        "validation_fraction": float(validation_fraction),
        "client_num_workers": int(num_workers),
        "simulation_num_threads": None if num_threads is None else int(num_threads),
        "gpu_config": gpu_config,
        "client_rng": "base_seed + site_index - 1",
        "torch_cudnn_benchmark": False,
        "torch_cudnn_deterministic": True,
        "final_evaluator": "common untouched CIFAR-10 test evaluator",
        "eval_batch_size": int(eval_batch_size),
        "eval_num_workers": int(eval_num_workers),
        "eval_device": eval_device,
    }


def method_run_config(
    method: str,
    *,
    fedprox_mu: float,
    fedce_mode: str,
    sample_exponent: float,
    representation_exponent: float,
    metric_prior_strength: float,
    max_blend_factor: float,
    activation_warmup_rounds: int,
    activation_patience: int,
    min_weight: float,
    max_weight: float,
    allow_changing_cohort_evidence: bool,
) -> dict:
    """Return all method-specific settings that affect one campaign method."""

    if method == "fedavg":
        return {
            "method": "fedavg",
            "aggregator": "native FedAvg weighted WEIGHT_DIFF",
        }
    if method == "fedopt":
        return {
            "method": "fedopt",
            "server_optimizer": "torch.optim.SGD",
            "server_lr": 1.0,
            "server_momentum": 0.6,
            "participation": "full-only reference",
        }
    if method == "fedprox":
        return {
            "method": "fedprox",
            "fedprox_mu": float(fedprox_mu),
        }
    if method == "scaffold":
        return {
            "method": "scaffold",
            "fedproxloss_mu": 0.0,
        }
    if method == "fedce":
        return {
            "method": "fedce",
            "fedce_mode": str(fedce_mode),
        }
    if method == "adaptive":
        return {
            "method": "adaptive",
            "sample_exponent": float(sample_exponent),
            "representation_exponent": float(representation_exponent),
            "quality_exponent": 0.0,
            "fairness_strength": 1.0,
            "metric_prior_strength": float(metric_prior_strength),
            "heterogeneity_threshold": 0.26,
            "heterogeneity_temperature": 0.04,
            "heterogeneity_deadband": 0.15,
            "performance_gap_threshold": 0.10,
            "performance_gap_temperature": 0.03,
            "performance_gap_deadband": 0.05,
            "max_blend_factor": float(max_blend_factor),
            "activation_warmup_rounds": int(activation_warmup_rounds),
            "activation_patience": int(activation_patience),
            "require_stable_cohort": not bool(allow_changing_cohort_evidence),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
        }
    raise ValueError(f"unsupported CIFAR-10 method {method!r}")


def condition_config(alpha: float, participation_rate: float, seed: int) -> dict:
    """Variables intentionally changed across experimental conditions."""

    return {
        "alpha": float(alpha),
        "participation_rate": float(participation_rate),
        "seed": int(seed),
    }
