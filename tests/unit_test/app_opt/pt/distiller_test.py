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

"""Shared session contracts with real optional backends in one interpreter.

Install nvidia-modelopt==0.48.0 and/or torchdistill==1.1.5 to select the real
backend cases. Normal unit runs still execute the native contract; absent
optional packages skip only their own cases. No data, GPU or runtime processes.
"""

import importlib.util
from copy import deepcopy
from functools import partial

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from nvflare.app_opt.pt.distillation import DistillationTrainer, kl_divergence
from nvflare.app_opt.pt.distillation.backends import ModelOptDistiller, TorchDistillDistiller


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        return torch.stack([model(inputs) for model in self.models]).mean(0)


def network():
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Tanh(), nn.Linear(6, 3))


def state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def hook_count(model):
    return sum(len(m._forward_hooks) for m in model.modules())


def require_backend(backend):
    package = {"modelopt": "modelopt", "torchdistill": "torchdistill"}.get(backend)
    if package and importlib.util.find_spec(package) is None:
        pytest.skip(f"optional {package} is not installed")


def make_session(backend, student, teacher, temperature):
    optimizer = partial(torch.optim.SGD, lr=0.01, momentum=0.9)
    if backend == "native":
        return DistillationTrainer(
            deepcopy(student),
            [teacher],
            optimizer,
            loss_fn=partial(kl_divergence, temperature=temperature),
            batch_adapter=lambda batch: batch[0],
        )
    if backend == "modelopt":
        return ModelOptDistiller(
            student, teacher, optimizer, temperature=temperature, batch_adapter=lambda batch: batch[0]
        )
    return TorchDistillDistiller(
        student,
        teacher,
        batch_adapter=lambda batch: batch[0],
        train_config={
            "teacher": {"forward_hook": {"output": ["models.0.0" if isinstance(teacher, Ensemble) else "0"]}},
            "student": {"forward_hook": {"output": ["0"]}},
            "criterion": {
                "key": "WeightedSumLoss",
                "kwargs": {
                    "sub_terms": {
                        "kd": {
                            "criterion": {
                                "key": "KDLoss",
                                "kwargs": {
                                    "student_module_path": ".",
                                    "student_module_io": "output",
                                    "teacher_module_path": ".",
                                    "teacher_module_io": "output",
                                    "temperature": temperature,
                                    "alpha": 0.0,
                                    "reduction": "batchmean",
                                },
                            },
                            "weight": temperature**2,
                        },
                    }
                },
            },
            "optimizer": {"key": "SGD", "kwargs": {"lr": 0.01, "momentum": 0.9}},
        },
    )


@pytest.mark.parametrize("backend", ["native", "modelopt", "torchdistill"])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("num_teachers", [1, 2])
def test_session_updates_export_and_cleanup(backend, temperature, num_teachers):
    require_backend(backend)
    torch.manual_seed(17)
    student = network()
    teacher = network() if num_teachers == 1 else Ensemble([network(), network()])
    teacher.train()
    before = state(teacher)
    before_modes = [m.training for m in teacher.modules()]
    before_gradients = [(p.requires_grad, p.grad) for p in teacher.parameters()]
    original_student = state(student)
    inputs = [torch.randn(8, 4), torch.randn(5, 4)]
    reference = make_session("native", student, teacher, temperature)
    session = make_session(backend, student, teacher, temperature)
    # Hold owned models only in the test to observe hook cleanup after close.
    owned_student = session.student if backend == "native" else session._student
    owned_teacher = session.teachers[0] if backend == "native" else session._teacher
    if backend != "native":
        assert hook_count(owned_student) > 0 and hook_count(owned_teacher) > 0
    try:
        for x in inputs:
            expected = reference.fit([(x, None)])
            actual = session.fit([(x, torch.full((len(x),), -999))])
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
            exported = session.export_student_state()
            assert set(exported) == set(student.state_dict())
            for name, value in exported.items():
                assert value.device.type == "cpu" and not value.requires_grad
                torch.testing.assert_close(value, reference.export_student_state()[name], rtol=0, atol=1e-7)
            name = next(iter(exported))
            retained = exported[name].clone()
            exported[name].fill_(999)
            torch.testing.assert_close(session.export_student_state()[name], retained, rtol=0, atol=0)
            # Next iteration must still train after snapshotting (no final export).
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert [m.training for m in teacher.modules()] == before_modes
        assert [(p.requires_grad, p.grad) for p in teacher.parameters()] == before_gradients
        for name, value in student.state_dict().items():
            torch.testing.assert_close(value, original_student[name], rtol=0, atol=0)
        assert hook_count(student) == hook_count(teacher) == 0
        with pytest.raises(ValueError, match="no batches"):
            session.fit([])
        with pytest.raises(ValueError, match="finite scalar"):
            session.fit([(torch.full((8, 4), float("nan")), None)])
    finally:
        reference.close()
        session.close()
    assert hook_count(owned_student) == hook_count(owned_teacher) == 0
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.export_student_state()
    with pytest.raises(RuntimeError, match="closed"):
        session.fit([(inputs[0], None)])


def test_native_export_is_separate_from_checkpoint_and_factories_can_replace_the_session():
    from nvflare.app_opt.pt.distillation import Distiller

    # Consumer touches only the protocol, including when a custom implementation
    # has no student/optimizer attributes at all.
    class Custom:
        def fit(self, batches):
            return {"steps": len(batches)}

        def export_student_state(self):
            return {"weight": torch.tensor([7.0])}

        def close(self):
            self.closed = True

    def consume(factory):
        session: Distiller = factory()
        try:
            session.fit([torch.ones(2, 4)])
            return session.export_student_state()
        finally:
            session.close()

    custom = Custom()
    assert consume(lambda: custom)["weight"].item() == 7 and custom.closed
    model = nn.Linear(4, 3)
    factory = partial(
        DistillationTrainer,
        student=model,
        teachers=[nn.Linear(4, 3)],
        optimizer_factory=partial(torch.optim.SGD, lr=0.01),
    )
    exported = consume(factory)
    assert set(exported) == set(model.state_dict())


def test_torchdistill_rejects_unsupported_training_semantics_before_optional_import():
    with pytest.raises(ValueError, match="accumulation"):
        TorchDistillDistiller(network(), network(), train_config={"optimizer": {"grad_accum_step": 2}})
    with pytest.raises(ValueError, match="single-stage"):
        TorchDistillDistiller(network(), network(), train_config={"scheduler": {}})
    with pytest.raises(ValueError, match="frozen"):
        TorchDistillDistiller(network(), network(), train_config={"teacher": {"requires_grad": True}})


def test_modelopt_supports_feature_loss_and_original_sequential_state():
    require_backend("modelopt")
    torch.manual_seed(21)
    student, teacher = network(), network()
    expected = deepcopy(student)
    inputs = torch.randn(8, 4)
    expected.train()
    expected(inputs)  # Match the complete forward's BatchNorm state update.
    loss = nn.functional.mse_loss(expected[0](inputs), teacher[0](inputs).detach())
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    session = ModelOptDistiller(
        student,
        teacher,
        partial(torch.optim.SGD, lr=0.01),
        criterion={("0", "0"): nn.MSELoss()},
    )
    try:
        metrics = session.fit([inputs])
        assert metrics["loss"] == pytest.approx(loss.item())
        for name, value in session.export_student_state().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=1e-7)
    finally:
        session.close()


@pytest.mark.parametrize("backend", ["modelopt", "torchdistill"])
def test_backend_constructor_failure_removes_registered_hooks(backend, monkeypatch):
    require_backend(backend)
    registered = []
    original = nn.Module.register_forward_hook

    def record(module, *args, **kwargs):
        handle = original(module, *args, **kwargs)
        registered.append((module, handle.id))
        return handle

    monkeypatch.setattr(nn.Module, "register_forward_hook", record)
    if backend == "modelopt":
        with pytest.raises(TypeError, match="optimizer"):
            ModelOptDistiller(network(), network(), lambda parameters: None)
    else:
        with pytest.raises(ValueError):
            TorchDistillDistiller(
                network(),
                network(),
                train_config={
                    "teacher": {"forward_hook": {"output": ["0"]}},
                    "student": {"forward_hook": {"output": ["0"]}},
                    "criterion": {"key": "WeightedSumLoss", "kwargs": {}},
                    "optimizer": {"key": "missing_optimizer", "kwargs": {}},
                },
            )
    assert registered
    assert all(identity not in module._forward_hooks for module, identity in registered)
