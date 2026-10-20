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

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = Path(__file__).parents[3] / "examples/advanced/cifar10/pt/src/task_checkpoint.py"
    spec = importlib.util.spec_from_file_location("cifar_task_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = {
        "job_id": "job",
        "site_name": "site-1",
        "task_checkpoint": {"input": None, "output": str(tmp_path / "pending")},
    }
    monkeypatch.setattr(module.flare, "system_info", lambda: info)
    return module, info


def _optimizer(scheduler_enabled=True):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8) if scheduler_enabled else None
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return optimizer, scheduler


@pytest.mark.parametrize("scheduler_enabled", [False, True])
@pytest.mark.parametrize("next_round", [1, 4])
def test_native_state_and_rng_roundtrip(checkpoint, scheduler_enabled, next_round):
    module, info = checkpoint
    random.seed(27)
    np.random.seed(27)
    torch.manual_seed(27)
    optimizer, scheduler = _optimizer(scheduler_enabled)
    module.save_training_state(optimizer, scheduler, 0)
    expected_random = random.random(), np.random.random(), torch.rand(5)
    info["task_checkpoint"]["input"] = info["task_checkpoint"]["output"]
    restored_optimizer, restored_scheduler = _optimizer(scheduler_enabled)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    module.restore_training_state(restored_optimizer, restored_scheduler, next_round)
    actual_random = random.random(), np.random.random(), torch.rand(5)
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for actual, expected in zip(restored_optimizer.state.values(), optimizer.state.values()):
        assert torch.equal(actual["momentum_buffer"], expected["momentum_buffer"])
    if scheduler_enabled:
        assert restored_scheduler.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize(
    "key,value", [("job_id", "other"), ("site_name", "site-2"), ("version", 2), ("round", 1), ("round", True)]
)
def test_rejects_wrong_scope_version_and_non_earlier_round(checkpoint, key, value):
    module, info = checkpoint
    optimizer, scheduler = _optimizer()
    module.save_training_state(optimizer, scheduler, 0)
    path = info["task_checkpoint"]["output"]
    state = torch.load(path, weights_only=True)
    state[key] = value
    torch.save(state, path)
    info["task_checkpoint"]["input"] = path
    with pytest.raises(RuntimeError, match="checkpoint"):
        module.restore_training_state(optimizer, scheduler, 1)


def test_rejects_changed_scheduler_configuration(checkpoint):
    module, info = checkpoint
    optimizer, scheduler = _optimizer()
    module.save_training_state(optimizer, scheduler, 0)
    info["task_checkpoint"]["input"] = info["task_checkpoint"]["output"]
    with pytest.raises(RuntimeError, match="scheduler configuration changed"):
        module.restore_training_state(optimizer, None, 1)


def test_job_lifetime_and_first_assignment_do_not_read_or_write_state(checkpoint, monkeypatch):
    module, info = checkpoint
    optimizer, scheduler = _optimizer()
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: pytest.fail("unexpected checkpoint read"))
    module.restore_training_state(optimizer, scheduler, 0)
    info.pop("task_checkpoint")
    monkeypatch.setattr(module.torch, "save", lambda *args, **kwargs: pytest.fail("unexpected checkpoint write"))
    module.restore_training_state(optimizer, scheduler, 1)
    module.save_training_state(optimizer, scheduler, 1)
