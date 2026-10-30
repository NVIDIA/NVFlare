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
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import client
import numpy as np
import pytest
from cohort import is_test
from evaluate import fit

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.workflows.lr.fedavg import FedAvgLR

Cohort = tuple[np.ndarray, np.ndarray, np.ndarray]


def site_cohorts(sizes: Sequence[int]) -> Mapping[str, Cohort]:
    rng = np.random.default_rng(0)
    cohorts = {}
    first_id = 0
    for number, size in enumerate(sizes, start=1):
        features = np.column_stack((rng.normal(size=size), rng.random((size, 3)) < [0.1, 0.3, 0.5]))
        logits = features @ np.array([[0.5], [1.2], [-0.7], [0.3]]) - 1.5
        labels = (rng.random((size, 1)) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
        cohorts[f"site-{number}"] = (np.arange(first_id, first_id + size), features.astype(np.float64), labels)
        first_id += size
    return cohorts


def client_update(monkeypatch: pytest.MonkeyPatch, site: str, cohort: Cohort, theta: np.ndarray) -> FLModel:
    sent = []
    flare = SimpleNamespace(
        init=Mock(),
        get_site_name=Mock(return_value=site),
        is_running=Mock(side_effect=[True, False]),
        receive=Mock(return_value=FLModel(params={NPConstants.NUMPY_KEY: theta.copy()}, current_round=0)),
        send=sent.append,
    )
    monkeypatch.setattr(client, "flare", flare)
    monkeypatch.setattr(client, "load_cohort", Mock(return_value=cohort))
    monkeypatch.setattr(sys, "argv", ["client.py", "--data_root", "unused"])
    client.main()
    return sent[0]


def test_federated_rounds_reach_pooled_fit_of_unequal_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    cohorts = site_cohorts((700, 350, 150))
    controller = FedAvgLR(damping_factor=0.8, n_features=4)
    theta = np.zeros((5, 1), dtype=np.float32)
    for _ in range(15):
        results = [client_update(monkeypatch, site, cohort, theta) for site, cohort in cohorts.items()]
        theta += controller.newton_raphson_aggregator_fn(results).params["newton_raphson_updates"]

    train = [~is_test(person_ids) for person_ids, _, _ in cohorts.values()]
    pooled = fit(
        np.vstack([features[keep] for keep, (_, features, _) in zip(train, cohorts.values())]),
        np.vstack([labels[keep] for keep, (_, _, labels) in zip(train, cohorts.values())]),
    )
    np.testing.assert_allclose(theta, pooled, atol=1e-4)
    assert 0.5 < results[0].metrics["auroc"] <= 1.0


def test_client_skips_auroc_when_held_out_patients_share_one_label(monkeypatch: pytest.MonkeyPatch) -> None:
    person_ids, features, labels = site_cohorts((200,))["site-1"]
    labels[is_test(person_ids)] = 0.0

    result = client_update(monkeypatch, "site-1", (person_ids, features, labels), np.zeros((5, 1), dtype=np.float32))

    assert not result.metrics
    assert result.params["hessian"].shape == (5, 5)
