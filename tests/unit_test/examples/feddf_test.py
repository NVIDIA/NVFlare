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

"""Essential example contracts: proxy isolation and FedDF fusion math."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def feddf(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "advanced"))
    state = np.random.get_state()
    try:
        yield lambda name: importlib.import_module(f"collab.pt_cifar10.feddf.{name}")
    finally:
        np.random.set_state(state)


def test_proxy_and_client_shards_are_disjoint_complete_and_repeatable(feddf, tmp_path):
    prepare = feddf("prepare_data").prepare_splits
    labels = np.tile(np.arange(10), 24)
    first, second = tmp_path / "first", tmp_path / "second"
    assert prepare(labels, first, 4, 0.1, 40, 7) == prepare(labels, second, 4, 0.1, 40, 7)
    proxy = np.load(first / "proxy.npy")
    shards = [np.load(first / f"site-{i}.npy") for i in range(1, 5)]
    assert len(proxy) == 40 and all(len(shard) >= 10 for shard in shards)
    combined = np.concatenate([proxy, *shards])
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(labels)))
    assert all(not np.intersect1d(proxy, shard).size for shard in shards)
    for path in first.glob("*.npy"):
        np.testing.assert_array_equal(np.load(path), np.load(second / path.name))


@pytest.mark.timeout(15)
def test_split_preparation_leaves_no_files_when_minimum_cannot_be_met(feddf, tmp_path):
    prepare = feddf("prepare_data").prepare_splits
    output = tmp_path / "split"
    # One class, eight sites, and just enough private examples under extreme skew.
    with pytest.raises(ValueError, match="could not give every site at least 10 samples"):
        prepare(np.zeros(100, dtype=np.int64), output, num_clients=8, proxy_size=20, alpha=0.01)
    assert not output.exists()


@pytest.mark.parametrize(("name", "value"), [("kd_lr", float("inf")), ("temperature", float("nan"))])
def test_factory_rejects_nonfinite_kd_settings(feddf, name, value):
    with pytest.raises(ValueError, match="must be finite and positive"):
        feddf("distiller").Cifar10DistillerFactory(**{name: value})


def test_seeded_client_supports_custom_site_names(feddf, monkeypatch, tmp_path):
    import torch

    client_module = importlib.import_module("collab.pt_cifar10.fedavg.client")
    collab = SimpleNamespace()
    monkeypatch.setattr(client_module, "collab", collab)
    calls = []

    def make_loader(**settings):
        calls.append(settings)
        return []

    monkeypatch.setattr(client_module, "make_data_loader", make_loader)
    with torch.random.fork_rng(devices=[]):
        for site_name in ("hospital-a", "hospital-a", "hospital-b", "site-1"):
            collab.site_name = site_name
            client = client_module.FedAvgClient(split_root=str(tmp_path), seed=10)
            client.device = torch.device("cpu")
            client.initialize()

    assert [call["site_name"] for call in calls] == ["hospital-a", "hospital-a", "hospital-b", "site-1"]
    assert all(call["split_root"] == str(tmp_path) for call in calls)
    first_seed, repeat_seed, other_seed, simulator_seed = [call["generator"].initial_seed() for call in calls]
    assert first_seed == repeat_seed and first_seed != other_seed
    assert simulator_seed == 11


def test_feddf_averages_then_distills_mean_logits_without_mutating_teachers(feddf, monkeypatch):
    from copy import deepcopy

    import torch
    from torch import nn

    from nvflare.app_opt.pt.distillation import kl_divergence
    from nvflare.collab.api.group_call_context import FrozenResultQueue

    server = feddf("server")

    def network():
        model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2))
        model.register_buffer("marker", torch.tensor(0))
        return model

    torch.manual_seed(7)
    monkeypatch.setattr(server, "SimpleNetwork", network)
    monkeypatch.setattr(server, "collab", SimpleNamespace(is_aborted=False))
    teachers = [network(), network()]
    teachers[-1].marker.fill_(9)
    states = [deepcopy(t.state_dict()) for t in teachers]
    results = FrozenResultQueue(
        [(f"site-{i + 1}", {"weights": state, "num_steps": i + 1}) for i, state in enumerate(states)], {}, 2
    )
    average = network()
    parameters = dict(average.named_parameters())
    average.load_state_dict(
        {
            name: (states[0][name] + states[1][name]) / 2 if name in parameters else states[-1][name]
            for name in states[0]
        }
    )
    model = network()  # Deliberately different from the client average.
    server.FedDFServer(aggregation="fedavg").aggregate(model, results, None, 0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, average.state_dict()[name], rtol=0, atol=0)

    inputs = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 1.0], [-2.0, -1.0]])
    expected = deepcopy(average)
    targets = torch.stack([t(inputs).detach() for t in teachers]).mean(0)
    loss = kl_divergence(expected(inputs), targets)
    mean_loss = sum(kl_divergence(expected(inputs), t(inputs).detach()) for t in teachers) / 2
    assert not torch.isclose(loss, mean_loss)
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model = network()
    factory = feddf("distiller").Cifar10DistillerFactory(kd_lr=0.01)
    metrics = server.FedDFServer(distiller_factory=factory).aggregate(
        model, results, [(inputs, torch.full((4,), -999))], 0
    )
    assert metrics["kd_steps"] == 1 and metrics["kd_samples"] == 4
    assert metrics["kd_loss"] == pytest.approx(loss.item())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), average.parameters()))
    for teacher, state in zip(teachers, states):
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure", [None, "backend", "empty"])
def test_server_uses_replaceable_session_export_and_always_closes(feddf, monkeypatch, failure):
    from copy import deepcopy

    import torch
    from torch import nn

    from nvflare.collab.api.group_call_context import FrozenResultQueue

    server = feddf("server")
    monkeypatch.setattr(server, "SimpleNetwork", lambda: nn.Linear(2, 2, bias=False))
    monkeypatch.setattr(server, "collab", SimpleNamespace(is_aborted=False))
    states = [{"weight": torch.full((2, 2), value)} for value in (1.0, 3.0)]
    results = FrozenResultQueue([(f"site-{i}", {"weights": weights}) for i, weights in enumerate(states)], {}, 2)
    captured = {}

    class Session:
        # Deliberately no student or optimizer attributes.
        def fit(self, batches):
            if failure == "backend":
                raise RuntimeError("backend failed")
            if failure == "empty":
                return {"loss": 0.0, "steps": 0, "samples": 0}
            return {"loss": 0.1, "steps": 1, "samples": 4}

        def export_student_state(self):
            captured["exported"] = True
            return {"weight": captured["average"]["weight"] + 0.5}

        def close(self):
            captured["closed"] = True

    def factory(student, teacher):
        captured["average"] = deepcopy(student.state_dict())
        return Session()

    model = nn.Linear(2, 2, bias=False)
    workflow = server.FedDFServer(distiller_factory=factory)
    if failure == "backend":
        with pytest.raises(RuntimeError, match="backend failed"):
            workflow.aggregate(model, results, [], 0)
    elif failure == "empty":
        with pytest.raises(ValueError, match="requires at least one proxy sample"):
            workflow.aggregate(model, results, [], 0)
    else:
        metrics = workflow.aggregate(model, results, [], 0)
        assert metrics["kd_steps"] == 1
        torch.testing.assert_close(model.weight, torch.full((2, 2), 2.5))
    torch.testing.assert_close(captured["average"]["weight"], torch.full((2, 2), 2.0))
    assert captured["closed"]
    assert captured.get("exported", False) == (failure is None)
