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

from copy import deepcopy
from functools import partial
from io import BytesIO

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from nvflare.app_opt.pt.distillation import DistillationTrainer, TeacherBuffer, kl_divergence


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        yield


def test_kl_matches_distribution_kl_and_only_student_gets_gradients():
    student = torch.tensor([[1.0, -2.0, 0.5], [0.1, 0.4, -0.2]], requires_grad=True)
    teacher = torch.tensor([[0.5, 2.0, -1.0], [-0.3, 0.2, 0.6]], requires_grad=True)
    temperature = 2.0
    expected = (
        torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=teacher.detach() / temperature),
            torch.distributions.Categorical(logits=student / temperature),
        )
        * temperature**2
    )

    loss = kl_divergence(student, teacher, temperature)
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(kl_divergence(student, teacher, temperature, "sum"), expected.sum())
    terms = kl_divergence(student, teacher, temperature, "none")
    assert terms.shape == student.shape
    torch.testing.assert_close(terms.sum(dim=1), expected)
    expected_gradient = torch.autograd.grad(expected.mean(), student)[0]
    loss.backward()
    torch.testing.assert_close(student.grad, expected_gradient)
    assert teacher.grad is None


def test_softened_probabilities_match_logits_and_invalid_contracts_fail():
    student, teacher = torch.randn(3, 4), torch.randn(3, 4)
    target = (teacher / 3).softmax(dim=1)
    torch.testing.assert_close(
        kl_divergence(student, target, 3, teacher_is_logits=False), kl_divergence(student, teacher, 3)
    )
    for kwargs in ({"temperature": 0}, {"temperature": float("nan")}, {"reduction": "mean"}):
        with pytest.raises(ValueError):
            kl_divergence(student, teacher, **kwargs)
    with pytest.raises(ValueError, match="matching shapes"):
        kl_divergence(student, teacher[:, :2])
    with pytest.raises(ValueError, match="probabilities"):
        kl_divergence(student, target * 2, teacher_is_logits=False)


def test_teacher_history_is_bounded_owned_and_restores_in_order():
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    buffer = TeacherBuffer(2)
    metadata = {"version": 0, "labels": ["a", "b", "c"]}
    for version in range(3):
        metadata["version"] = version
        with torch.no_grad():
            model[0].weight.fill_(version)
        buffer.add(model, metadata)
    metadata["labels"].clear()
    with torch.no_grad():
        model[0].weight.fill_(99)

    snapshots = list(buffer)
    assert len(snapshots) == 2
    assert [item.metadata["version"] for item in snapshots] == [2, 1]
    assert snapshots[0].metadata["labels"] == ["a", "b", "c"]
    assert snapshots[0].model_state["0.weight"].mean().item() == 2
    assert snapshots[0].model_state["0.weight"].device.type == "cpu"
    assert hasattr(snapshots[0].model_state, "_metadata")
    exported = buffer.state_dict()
    restored = TeacherBuffer(2)
    restored.load_state_dict(exported)
    exported["teachers"][0]["model_state"]["0.weight"].zero_()
    assert next(iter(restored)).model_state["0.weight"].mean().item() == 2
    assert next(iter(buffer)).model_state["0.weight"].mean().item() == 2
    with pytest.raises(ValueError, match="capacity"):
        TeacherBuffer(1).load_state_dict(buffer.state_dict())


def test_heterogeneous_kd_reduces_loss_without_changing_original_teachers():
    teacher = nn.Sequential(nn.Linear(3, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Dropout(), nn.Linear(6, 2))
    teacher.train()
    original = deepcopy(teacher.state_dict())
    inputs = torch.randn(16, 3)
    trainer = DistillationTrainer(
        lambda: nn.Linear(3, 2),
        [teacher, lambda: nn.Linear(3, 2)],
        partial(torch.optim.SGD, lr=0.5),
        loss_fn=partial(kl_divergence, temperature=2),
    )

    def objective():
        with torch.no_grad():
            return sum(trainer.loss_fn(trainer.student(inputs), model(inputs)) for model in trainer.teachers) / 2

    before = objective()
    metrics = trainer.fit([inputs] * 30)
    assert objective() < before * 0.5
    assert metrics["steps"] == trainer.num_steps == 30
    assert metrics["samples"] == 16 * 30
    assert teacher.training
    assert all(parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[name])
    for model in trainer.teachers:
        assert not model.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())


class _OutputModel(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return {self.key: self.linear(inputs)}


def test_batch_and_output_adapters_respect_the_step_budget():
    batch = (torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
    trainer = DistillationTrainer(
        _OutputModel("student"),
        [_OutputModel("teacher")],
        partial(torch.optim.SGD, lr=0.1),
        batch_adapter=lambda item: item[0],
        student_output_adapter=lambda value: value["student"],
        teacher_output_adapter=lambda value: value["teacher"],
    )
    batches = iter([batch] * 3)
    metrics = trainer.fit(batches, max_steps=2)
    assert metrics["steps"] == 2
    assert next(batches) is batch


def test_checkpoint_restores_optimizer_teachers_and_caller_owned_rng():
    def make_trainer():
        return DistillationTrainer(
            lambda: nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Dropout(0.3), nn.Linear(5, 2)),
            [lambda: nn.Linear(3, 2)],
            partial(torch.optim.SGD, lr=0.2, momentum=0.9),
        )

    trainer = make_trainer()
    inputs = torch.randn(8, 3)
    trainer.fit([inputs] * 3)
    stream = BytesIO()
    torch.save(trainer.state_dict(), stream)
    rng = torch.random.get_rng_state()
    trainer.fit([inputs] * 2)
    expected = deepcopy(trainer.student.state_dict())

    restored = make_trainer()
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    # Constructors can consume RNG. Restore it after model/state construction.
    torch.random.set_rng_state(rng)
    restored.fit([inputs] * 2)
    assert restored.num_steps == trainer.num_steps == 5
    for name, value in restored.student.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_empty_inputs_and_nonfinite_loss_do_not_step_the_optimizer():
    student = nn.Linear(3, 2)
    trainer = DistillationTrainer(student, [nn.Linear(3, 2)], partial(torch.optim.SGD, lr=0.1))
    with pytest.raises(ValueError, match="no batches"):
        trainer.fit([])
    trainer.loss_fn = lambda logits, target: logits.sum() * float("nan")
    original = deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="finite scalar"):
        trainer.fit([torch.ones(4, 3)])
    assert trainer.num_steps == 0
    for name, value in student.state_dict().items():
        torch.testing.assert_close(value, original[name])
    with pytest.raises(ValueError, match="teacher"):
        DistillationTrainer(student, [], partial(torch.optim.SGD, lr=0.1))
