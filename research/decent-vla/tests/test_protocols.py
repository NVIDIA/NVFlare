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

"""Verify that protocols are correctly implemented by reference adapters/strategies."""
from __future__ import annotations

import torch

from decent_vla.adapters._mock.policy import MockPolicy
from decent_vla.benchmarks._mock import MockBenchmark
from decent_vla.core.backend import Backend
from decent_vla.core.benchmark import Benchmark
from decent_vla.core.policy import Policy
from decent_vla.core.strategy import Strategy, StrategyBase
from decent_vla.strategies import FedAdam, FedAvg, FedProx


def test_mock_backend_satisfies_protocol(backend):
    assert isinstance(backend, Backend)


def test_mock_backend_methods_run(backend, data_cfg):
    loader = backend.build_dataloader(data_cfg, partition_indices=None)
    batch = next(loader)
    out = backend.compute_loss(batch)
    assert "loss" in out
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].requires_grad


def test_mock_backend_get_set_state_roundtrip(backend):
    s1 = backend.get_state("shared")
    # Perturb, then restore.
    perturbed = {k: v + 0.1 for k, v in s1.items()}
    backend.set_state(perturbed, scope="shared")
    s2 = backend.get_state("shared")
    for k in s1:
        assert torch.allclose(s2[k], perturbed[k])
    backend.set_state(s1, scope="shared")
    s3 = backend.get_state("shared")
    for k in s1:
        assert torch.allclose(s3[k], s1[k])


def test_mock_backend_personal_scope_is_empty(backend):
    assert backend.get_state("personal") == {}
    assert backend.supports_personal_params is False


def test_mock_policy_satisfies_protocol(backend):
    policy = backend.build_policy()
    assert isinstance(policy, Policy)
    assert isinstance(policy, MockPolicy)


def test_mock_benchmark_satisfies_protocol():
    from decent_vla.configs.schemas import BenchmarkConfig

    bench = MockBenchmark.from_config(BenchmarkConfig(name="mock", suites=["default"]))
    assert isinstance(bench, Benchmark)


def test_strategies_satisfy_protocol():
    for cls in (FedAvg, FedAdam, FedProx):
        s = cls()
        assert isinstance(s, Strategy), f"{cls.__name__} does not satisfy Strategy protocol"
        assert isinstance(s, StrategyBase)


def test_strategy_defaults_are_fedavg_semantics():
    """The default base class IS FedAvg — same aggregate, no loss modification."""
    base = StrategyBase()
    avg = FedAvg()
    assert avg.aggregate.__func__ is base.aggregate.__func__
    assert avg.modify_loss.__func__ is base.modify_loss.__func__
