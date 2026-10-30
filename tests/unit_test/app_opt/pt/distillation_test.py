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


def test_synthetic_pool_bounds_ownership_custom_sampling_and_state():
    from nvflare.app_opt.pt.distillation.pool import SyntheticPool

    pool = SyntheticPool(3)
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    targets = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    pool.add(inputs, targets)
    inputs.zero_()
    targets.zero_()
    samples, values = pool.sample(3, sampler=lambda x, y, n, rng: torch.arange(n))
    assert samples.tolist() == [[4, 5], [6, 7], [8, 9]]
    assert values.flatten().tolist() == [2, 3, 4]
    samples.zero_()
    exported = pool.state_dict()
    restored = SyntheticPool(3)
    restored.load_state_dict(exported)
    exported["inputs"].zero_()
    torch.testing.assert_close(restored.state_dict()["inputs"], pool.state_dict()["inputs"])
    first = pool.sample(8, generator=torch.Generator().manual_seed(9))
    second = restored.sample(8, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(first[0], second[0])
    with pytest.raises(ValueError, match="target presence"):
        pool.add(torch.ones(1, 2))
    with pytest.raises(ValueError, match="empty"):
        SyntheticPool(3).sample(1)
    with pytest.raises(ValueError, match="indices"):
        pool.sample(1, sampler=lambda *args: torch.tensor([3]))


def test_class_sampling_and_probe_probabilities_have_no_fixed_class_count():
    from nvflare.app_opt.pt.distillation.classification import mean_class_probabilities, sample_by_class

    targets = torch.tensor([0, 0, 2])
    rng = torch.Generator().manual_seed(9)
    actual = sample_by_class(torch.zeros(3, 2), targets, 16, rng, probabilities=torch.tensor([0.1, 0.2, 0.3, 0.4]))
    # Missing-class mass 0.6 is shared uniformly: [0.05+0.2, 0.05+0.2, 0.3+0.2].
    expected = torch.multinomial(
        torch.tensor([0.25, 0.25, 0.5]), 16, replacement=True, generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(actual, expected)
    model = _OutputModel("teacher").train()
    inputs = torch.randn(5, 3)
    original = deepcopy(model.state_dict())
    reference = deepcopy(model).eval()
    expected = (reference(inputs)["teacher"] / 2).softmax(1).mean(0)
    probabilities = mean_class_probabilities(
        model, [inputs[:2], inputs[2:]], temperature=2, output_adapter=lambda output: output["teacher"]
    )
    torch.testing.assert_close(probabilities, expected.detach())
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_synthesis_defaults_train_only_generator_and_return_model_ready_inputs():
    from nvflare.app_opt.pt.distillation.synthesis import DataFreeSynthesizer

    teacher = nn.Sequential(nn.Flatten(), nn.Linear(1 * 8 * 12, 4)).train()
    original = deepcopy(teacher.state_dict())
    synthesis = DataFreeSynthesizer(
        [teacher], input_shape=(1, 8, 12), latent_size=4, rng=torch.Generator().manual_seed(3)
    )
    initial = deepcopy(synthesis.generator.state_dict())
    inputs, metrics = synthesis.synthesize(batch_size=2, steps=2, targets=torch.tensor([0, 3]))
    assert inputs.shape == (2, 1, 8, 12)
    assert inputs.device.type == "cpu" and not inputs.requires_grad
    assert torch.isfinite(inputs).all() and 0 <= inputs.min() <= inputs.max() <= 1
    assert metrics["steps"] == synthesis.num_steps == 2
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(synthesis.generator.state_dict()["project.weight"], initial["project.weight"])
    assert teacher.training
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert all(value.grad is None and not value.requires_grad for value in synthesis.teachers[0].parameters())
    assert all(value.grad is None for value in teacher.parameters())


def test_custom_regression_synthesis_keeps_input_gradients_and_feeds_existing_kd():
    from nvflare.app_opt.pt.distillation.pool import SyntheticPool
    from nvflare.app_opt.pt.distillation.synthesis import DataFreeSynthesizer

    teacher = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        teacher.weight.fill_(0.5)
    noise = torch.ones(4, 2)

    def objective(inputs, teachers, student, targets):
        assert student is None
        return (teachers[0](inputs) - targets).square().mean()

    synthesis = DataFreeSynthesizer(
        [teacher],
        generator=nn.Linear(2, 3),
        objective=objective,
        optimizer_factory=partial(torch.optim.SGD, lr=0.2),
        transform=lambda value: value * 0.5,
        latent_sampler=lambda count, rng: noise,
    )
    targets = torch.zeros(4, 1)
    before = objective(synthesis.transform(synthesis.generator(noise)), synthesis.teachers, None, targets).item()
    inputs, metrics = synthesis.synthesize(4, 30, targets)
    assert metrics["loss"] < before * 0.01
    assert inputs.shape == (4, 3)
    pool = SyntheticPool(8)
    pool.add(inputs, targets)
    trainer = DistillationTrainer(nn.Linear(3, 1), [teacher], partial(torch.optim.SGD, lr=0.1), loss_fn=nn.MSELoss())
    assert trainer.fit([pool.sample(4)[0]] * 2)["steps"] == 2


def test_feature_adapters_and_adversarial_term_preserve_input_gradients_and_remove_hooks():
    from nvflare.app_opt.pt.distillation.classification import ClassificationObjective, adversarial_disagreement
    from nvflare.app_opt.pt.distillation.features import forward_with_feature_statistics

    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2)).eval().requires_grad_(False)
    inputs = torch.randn(6, 3, requires_grad=True)
    _, loss = forward_with_feature_statistics(model, inputs)
    expected = inputs.mean(0).norm() + (inputs.var(0, unbiased=False) - 1).norm()
    torch.testing.assert_close(loss, expected)
    assert torch.autograd.grad(loss, inputs)[0].abs().sum() > 0
    assert not model[0]._forward_pre_hooks
    custom = ClassificationObjective(
        confidence_weight=0,
        diversity_weight=0,
        feature_weight=1,
        feature_adapter=lambda teacher, values: (teacher(values), values.mean().square()),
    )
    torch.testing.assert_close(custom(inputs, [model]), inputs.mean().square())
    teacher_logits = torch.randn(3, 2, requires_grad=True)
    student_logits = torch.randn(3, 2, requires_grad=True)
    loss = adversarial_disagreement([teacher_logits], student_logits, only_agree=False)
    assert all(value.abs().sum() > 0 for value in torch.autograd.grad(loss, (teacher_logits, student_logits)))

    teacher_logits = torch.tensor([[60.0, -60.0]], requires_grad=True)
    student_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    loss = adversarial_disagreement([teacher_logits], student_logits, only_agree=False)
    assert all(torch.isfinite(value).all() for value in torch.autograd.grad(loss, (teacher_logits, student_logits)))

    class Fail(nn.Module):
        def forward(self, value):
            raise RuntimeError("forward failed")

    failing = nn.Sequential(nn.BatchNorm1d(3), Fail()).eval()
    with pytest.raises(RuntimeError, match="forward failed"):
        forward_with_feature_statistics(failing, inputs)
    assert not failing[0]._forward_pre_hooks


def test_synthesis_checkpoint_restores_optimizer_and_external_rng_continuation():
    from nvflare.app_opt.pt.distillation.synthesis import DataFreeSynthesizer

    def make_synthesis():
        return DataFreeSynthesizer(
            [nn.Linear(3, 1)],
            generator=nn.Linear(2, 3),
            objective=lambda inputs, teachers, student, targets: teachers[0](inputs).square().mean(),
            transform=nn.Identity(),
            latent_size=2,
            rng=torch.Generator().manual_seed(4),
        )

    synthesis = make_synthesis()
    synthesis.synthesize(4, 2)
    stream = BytesIO()
    torch.save(synthesis.state_dict(), stream)
    rng = synthesis.rng.get_state()
    expected, metrics = synthesis.synthesize(4, 3)
    restored = make_synthesis()
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.rng.set_state(rng)
    actual, resumed = restored.synthesize(4, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metrics == resumed
    assert restored.num_steps == synthesis.num_steps == 5
    for name, value in restored.generator.state_dict().items():
        torch.testing.assert_close(value, synthesis.generator.state_dict()[name], rtol=0, atol=0)


def test_cpu_sampling_and_synthesis_ignore_the_process_default_device():
    from nvflare.app_opt.pt.distillation.classification import sample_by_class
    from nvflare.app_opt.pt.distillation.image import ImageTransform
    from nvflare.app_opt.pt.distillation.pool import SyntheticPool
    from nvflare.app_opt.pt.distillation.synthesis import DataFreeSynthesizer

    images = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    targets = torch.tensor([0, 1, 0, 1])
    pool = SyntheticPool(4)
    pool.add(images, targets)
    for rng in (None, torch.Generator(device="cpu").manual_seed(3)):
        synthesis = DataFreeSynthesizer(
            [nn.Sequential(nn.Flatten(), nn.Linear(16, 2))], input_shape=(1, 4, 4), latent_size=2, rng=rng
        )
        # Meta catches implicit device allocations without requiring a CUDA runner.
        with torch.device("meta"):
            samples, labels = pool.sample(8, generator=rng)
            balanced, _ = pool.sample(8, sampler=partial(sample_by_class, probabilities=[0.4, 0.6]), generator=rng)
            transformed = ImageTransform(mean=[0.5], std=[0.25], generator=rng)(images)
            generated, metrics = synthesis.synthesize(2, 1)
        assert samples.device.type == labels.device.type == balanced.device.type == "cpu"
        indices = (samples[:, 0, 0, 0] * 4).long()
        assert torch.equal(samples, images[indices]) and torch.equal(labels, targets[indices])
        assert transformed.device.type == generated.device.type == "cpu"
        assert transformed.shape == images.shape and generated.shape == (2, 1, 4, 4)
        assert torch.isfinite(transformed).all() and torch.isfinite(generated).all()
        assert metrics["steps"] == synthesis.num_steps == 1
