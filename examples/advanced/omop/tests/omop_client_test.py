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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import client
import numpy as np
import pytest
from evaluate import fit

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR


def client_update(
    monkeypatch: pytest.MonkeyPatch, features: np.ndarray, labels: np.ndarray, theta: np.ndarray
) -> FLModel:
    sent = []
    flare = SimpleNamespace(
        init=Mock(),
        get_site_name=Mock(return_value="site-1"),
        is_running=Mock(side_effect=[True, False]),
        receive=Mock(return_value=FLModel(params={NPConstants.NUMPY_KEY: theta.copy()})),
        send=sent.append,
    )
    monkeypatch.setattr(client, "flare", flare)
    monkeypatch.setattr(client, "load_cohort", Mock(return_value=(np.arange(len(labels)), features, labels)))
    monkeypatch.setattr(sys, "argv", ["client.py", "--data_root", "unused"])
    client.main()
    return sent[0]


def test_federated_rounds_reach_pooled_fit_of_unequal_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    sites = []
    for size in (700, 350, 150):
        features = np.column_stack((rng.normal(size=size), rng.random((size, 3)) < [0.1, 0.3, 0.5])).astype(float)
        logits = features @ np.array([[0.5], [1.2], [-0.7], [0.3]]) - 1.5
        sites.append((features, (rng.random((size, 1)) < 1.0 / (1.0 + np.exp(-logits))).astype(float)))

    controller = FedAvgLR(damping_factor=0.8, n_features=4)
    theta = np.zeros((5, 1), dtype=np.float32)
    for _ in range(15):
        results = [client_update(monkeypatch, features, labels, theta) for features, labels in sites]
        theta += controller.newton_raphson_aggregator_fn(results).params["newton_raphson_updates"]

    pooled = fit(np.vstack([x for x, _ in sites]), np.vstack([y for _, y in sites]))
    np.testing.assert_allclose(theta, pooled, atol=1e-4)
