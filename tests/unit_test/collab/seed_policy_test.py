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

"""Stable seeds, isolated streams, and repeated stochastic training checks."""

import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nvflare.apis.signal import Signal
from nvflare.collab import collab
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.seed_policy import SeededTrainArgs, SeedPolicy


def test_seed_derivation_v1_reference_vectors():
    policy = SeedPolicy(10)
    assert policy.seed("train", assignment_id=0, logical_name="alice") == 458068982632743392
    assert policy.seed("train", assignment_id=1, logical_name="alice") == 5706884726468335921
    assert policy.seed("kd") == 6729962603077186818
    assert (
        len(
            {
                policy.seed("train", assignment_id=0, logical_name="alice"),
                policy.seed("train", assignment_id=0, logical_name="bob"),
                policy.seed("data", assignment_id=0, logical_name="alice"),
                SeedPolicy(11).seed("train", assignment_id=0, logical_name="alice"),
            }
        )
        == 4
    )


def test_seed_is_stable_across_processes_python_hash_seeds_and_unicode():
    code = "from nvflare.collab.seed_policy import SeedPolicy; print(SeedPolicy(10).seed('train', assignment_id=7, logical_name='客户|a'))"
    observed = [
        subprocess.check_output([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=value), text=True)
        for value in ("1", "90210")
    ]
    assert observed[0] == observed[1]
    assert int(observed[0]) == SeedPolicy(10).seed("train", assignment_id=7, logical_name="客户|a")


def test_generator_streams_are_independent_and_leave_global_state_untouched():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    policy = SeedPolicy(10)
    first = policy.numpy_rng("train")
    reference = policy.numpy_rng("train")
    policy.numpy_rng("kd").normal(size=1000)
    policy.numpy_rng("synthesis").normal(size=100)
    policy.python_rng("probe").random()
    np.testing.assert_array_equal(first.normal(size=8), reference.normal(size=8))
    assert policy.python_rng("train").random() == policy.python_rng("train").random()
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**63])
def test_invalid_root_seeds_fail_early(seed):
    with pytest.raises(ValueError, match="seed"):
        SeedPolicy(seed)


@pytest.mark.parametrize(
    "scope",
    [
        {"stream": ""},
        {"stream": "train", "assignment_id": -1},
        {"stream": "train", "assignment_id": True},
        {"stream": "train", "logical_name": ""},
    ],
)
def test_invalid_stream_identities_fail_early(scope):
    with pytest.raises(ValueError):
        SeedPolicy(10).seed(**scope)


def _call(worker, metadata=None):
    return SimpleNamespace(
        client_name=worker,
        assignment=SimpleNamespace(
            assignment_id=5, logical_name="logical-a", metadata={} if metadata is None else metadata
        ),
    )


def test_train_adapter_uses_logical_identity_and_records_or_preserves_explicit_seeds():
    adapter = SeededTrainArgs(SeedPolicy(10))
    first, second = _call("site-1"), _call("site-99")
    seed = adapter(first)["train_seeds"]["site-1"]
    assert adapter(second)["train_seeds"]["site-99"] == seed
    assert first.assignment.metadata["train_seed"] == seed
    explicit = _call("site-2", {"train_seed": 123})
    assert adapter(explicit)["train_seeds"]["site-2"] == 123
    assert explicit.assignment.metadata["train_seed"] == 123
    with pytest.raises(ValueError):
        adapter(_call("site-1", {"train_seed": -1}))


def test_policy_and_train_adapter_import_without_tensor_frameworks():
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torchvision', 'kornia'):
            raise ImportError('optional dependency blocked')
sys.meta_path.insert(0, Block())
from nvflare.collab.seed_policy import SeedPolicy, SeededTrainArgs
assert SeedPolicy(10).seed('train') >= 0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("fail", [False, True])
def test_torch_scope_repeats_draws_and_restores_rngs_even_after_exceptions(fail):
    torch = pytest.importorskip("torch")
    from nvflare.collab.pt_seed import torch_seed_scope

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    draws = []
    for _ in range(2):
        try:
            with torch_seed_scope(123):
                draws.append((random.random(), float(np.random.random()), torch.rand(4)))
                if fail:
                    raise RuntimeError("training failed")
        except RuntimeError:
            assert fail
        assert random.getstate() == python_state
        assert np.random.get_state()[2:] == numpy_state[2:]
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert torch.equal(torch.get_rng_state(), torch_state)
    assert draws[0][:2] == draws[1][:2] and torch.equal(draws[0][2], draws[1][2])


def test_cpu_scope_does_not_initialize_or_seed_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from nvflare.collab.pt_seed import torch_seed_scope

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU scope touched CUDA")

    for name in ("manual_seed", "manual_seed_all", "get_rng_state", "set_rng_state", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    with torch_seed_scope(123):
        assert torch.rand(1).numel() == 1


class _RandomSamples:
    """Map-style dataset using all three RNGs; also picklable for spawn workers."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        import torch

        return torch.tensor([index / 8, random.random(), float(np.random.random()), torch.rand(1).item()]), index % 2


@pytest.fixture
def cifar_modules(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    from nvflare.collab.pt_seed import torch_seed_scope

    folder = Path(__file__).resolve().parents[3] / "examples/advanced/collab/pt_async_cifar10"
    names = ("job", "async_aggregator", "model", "trainer", "cifar10_data")
    saved = {name: sys.modules.pop(name, None) for name in names}
    monkeypatch.syspath_prepend(str(folder))
    previous = get_call_context()
    try:
        trainer = importlib.import_module("trainer")

        def tiny_model(seed=42):
            with torch_seed_scope(seed):
                return torch.nn.Sequential(
                    torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Dropout(0.4), torch.nn.Linear(8, 2)
                )

        monkeypatch.setattr(trainer, "ModerateCNN", tiny_model)
        monkeypatch.setattr(trainer.datasets, "CIFAR10", lambda **kwargs: _RandomSamples())
        yield trainer
    finally:
        set_call_context(previous)
        for name, previous_module in saved.items():
            sys.modules.pop(name, None)
            if previous_module is not None:
                sys.modules[name] = previous_module


def _training_run(module, root, run_seed, noise_seed, workers):
    import torch

    trainer = module.Cifar10Trainer(
        data_root=str(root),
        train_idx_root=str(root),
        local_batch_size=2,
        aggregation_epochs=1,
        total_client_rounds=3,
        device="cpu",
        num_workers=workers,
    )
    app = ClientApp(trainer)
    app.name = "site-1"
    app._abort_signal = Signal()
    app.new_context("server", "site-1")
    trainer.initialize()
    optimizer = trainer._optimizer
    state = module.get_model_params(trainer._local_model, target_device="cpu")
    policy = SeedPolicy(run_seed)
    losses = []
    for identity in range(2):
        # Ambient draws differ between runs; job-local training should not.
        random.seed(noise_seed + identity)
        np.random.seed(noise_seed + identity)
        torch.manual_seed(noise_seed + identity)
        random.random(), np.random.random(), torch.rand(100)
        seed = policy.seed("train", assignment_id=identity, logical_name="site-1")
        state, metrics, metadata = trainer.train(
            {"site-1": identity}, identity, state, {"site-1": "site-1"}, {"site-1": seed}
        )
        assert metadata["train_seed"] == seed
        assert trainer._optimizer is optimizer
        losses.append(metrics["train_loss"])
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), losses


@pytest.mark.parametrize("workers", [0, 1])
def test_repeated_stochastic_training_has_identical_model_hash_with_fixed_assignment_seeds(
    cifar_modules, tmp_path, workers
):
    np.save(tmp_path / "site-1.npy", np.arange(8))
    first = _training_run(cifar_modules, tmp_path, 10, 100, workers)
    second = _training_run(cifar_modules, tmp_path, 10, 999, workers)
    assert first == second
    if workers == 0:
        assert _training_run(cifar_modules, tmp_path, 11, 100, workers)[0] != first[0]


def test_cifar_job_exports_seed_policy_and_replay_through_public_constructor(cifar_modules, tmp_path):
    module = importlib.import_module("job")
    args = module.define_parser().parse_args(
        [
            "--num-clients",
            "2",
            "--num-rounds",
            "2",
            "--buffer-size",
            "1",
            "--arrival-sequence",
            "1",
            "0",
            "--run-seed",
            "123",
        ]
    )
    args.data_root = str(tmp_path)
    args.train_idx_root = str(tmp_path)
    recipe = module.make_recipe(args)
    assert recipe.server.train_kwargs.seed_policy.run_seed == 123
    assert recipe.server.scheduler.arrival_sequence == [1, 0]
    recipe.export(str(tmp_path / "export"))
    config = json.loads(next((tmp_path / "export").rglob("config_fed_server.json")).read_text())
    server = next(item for item in config["components"] if item["id"] == "_server")
    from nvflare.fuel.utils.component_builder import ComponentBuilder

    class Builder(ComponentBuilder):
        def get_module_scanner(self):
            return None

    restored = Builder().build_component(server)
    assert isinstance(restored.train_kwargs, SeededTrainArgs)
    assert isinstance(restored.train_kwargs.seed_policy, SeedPolicy)
    assert restored.train_kwargs(_call("site-1")) == recipe.server.train_kwargs(_call("site-1"))


class _StochasticClient:
    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds):
        import torch

        from nvflare.collab.pt_seed import torch_seed_scope

        site = collab.site_name
        seed = train_seeds[site]
        with torch_seed_scope(seed):
            parameter = torch.nn.Parameter(torch.from_numpy(global_model["w"].copy()))
            optimizer = torch.optim.SGD([parameter], lr=0.05)
            for _ in range(3):
                inputs = torch.randn(8) + float(np.random.random()) + random.random()
                loss = (parameter * inputs - torch.randn(8)).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            updated = parameter.detach().numpy().copy()
        return (
            {"w": updated},
            {},
            {
                "assignment_id": assignment_ids[site],
                "model_version": model_version,
                "logical_name": logical_assignments[site],
                "train_seed": seed,
            },
        )


def test_seeded_stochastic_async_replay_matches_model_hash_across_physical_worker_counts():
    pytest.importorskip("torch")
    from nvflare.collab.async_scheduling import ReplayScheduler

    from .async_extensions_test import _TraceApplication
    from .async_recipe_test import _execute

    runs = []
    for workers in (1, 2):
        application = _TraceApplication()
        output = _execute(
            application,
            [_StochasticClient() for _ in range(workers)],
            num_updates=4,
            num_active_jobs=3,
            scheduler=ReplayScheduler([2, 0, 1, 3], ["alice", "bob", "alice", "carol"]),
            train_kwargs=SeededTrainArgs(SeedPolicy(10)),
        )
        trace = [
            (event["assignment_id"], event["logical_name"], event["base_version"], event["metadata"]["train_seed"])
            for event in application.events
            if event["phase"] == "accept"
        ]
        runs.append((hashlib.sha256(output["w"].tobytes()).hexdigest(), trace))
    assert runs[0] == runs[1]
    assert [event[0] for event in runs[0][1]] == [2, 0, 1, 3]
