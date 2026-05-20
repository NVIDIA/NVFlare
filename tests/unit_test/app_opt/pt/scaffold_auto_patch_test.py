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

import threading

import pytest
import torch
from torch import nn

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper
from nvflare.app_opt.pt.scaffold_auto_patch import get_pt_scaffold_auto_patch_manager


@pytest.fixture(autouse=True)
def clean_auto_patch_manager():
    manager = get_pt_scaffold_auto_patch_manager()
    manager.disable()
    yield manager
    manager.disable()


def _model(weight=1.0):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def _state_dict(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _state_like(model, value):
    return {name: torch.full_like(param, value) for name, param in model.state_dict().items()}


def _input_model(model, ctrl_value=0.5):
    return FLModel(
        params=_state_dict(model),
        current_round=0,
        meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _state_like(model, ctrl_value)},
    )


def _sgd_step(model, optimizer, grad_value=1.0):
    optimizer.zero_grad()
    model.weight.grad = torch.full_like(model.weight, grad_value)
    optimizer.step()


def test_fedavg_style_supervised_loop_gets_scaffold_controls(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    input_model = _input_model(model, ctrl_value=0.5)
    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)

    features = torch.tensor([[1.0]])
    labels = torch.tensor([[0.5]])
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(features), labels)
    loss.backward()
    optimizer.step()

    assert torch.allclose(model.weight, torch.tensor([[0.85]]))

    output_model = FLModel(params=_state_dict(model), metrics={"loss": loss.item()}, meta={"user_metric": "kept"})
    manager.on_send(output_model)

    assert output_model.metrics["loss"] == loss.item()
    assert output_model.meta["user_metric"] == "kept"
    assert output_model.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] == 1
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta


def test_local_controls_persist_across_rounds(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    first_input = _input_model(model, ctrl_value=0.5)
    manager.on_receive(first_input, task_name="train", train_task_name="train")
    model.load_state_dict(first_input.params)
    _sgd_step(model, optimizer)
    manager.on_send(FLModel(params=_state_dict(model)))

    second_input = _input_model(_model(weight=1.0), ctrl_value=0.5)
    manager.on_receive(second_input, task_name="train", train_task_name="train")
    model.load_state_dict(second_input.params)
    _sgd_step(model, optimizer)

    assert torch.allclose(model.weight, torch.tensor([[0.95]]))


def test_preserves_user_step_count_when_present(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    input_model = _input_model(model, ctrl_value=0.5)
    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)
    _sgd_step(model, optimizer)

    output_model = FLModel(params=_state_dict(model), meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: 7})
    manager.on_send(output_model)

    assert output_model.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] == 7
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta


def test_manual_helper_calls_fail_in_auto_mode(clean_auto_patch_manager):
    clean_auto_patch_manager.enable()
    helper = PTScaffoldHelper()

    with pytest.raises(RuntimeError, match="auto_scaffold=True"):
        helper.model_update()

    with pytest.raises(RuntimeError, match="auto_scaffold=True"):
        helper.terms_update()


def test_hooks_are_idempotent_and_restore_original_methods(clean_auto_patch_manager):
    original_load_state_dict = torch.nn.Module.load_state_dict
    original_sgd_step = torch.optim.SGD.step

    clean_auto_patch_manager.enable()
    patched_load_state_dict = torch.nn.Module.load_state_dict
    patched_sgd_step = torch.optim.SGD.step
    clean_auto_patch_manager.enable()

    assert torch.nn.Module.load_state_dict is patched_load_state_dict
    assert torch.optim.SGD.step is patched_sgd_step

    clean_auto_patch_manager.disable()

    assert torch.nn.Module.load_state_dict is original_load_state_dict
    assert torch.optim.SGD.step is original_sgd_step


def test_auto_patch_state_is_thread_local_for_concurrent_clients(clean_auto_patch_manager):
    barrier = threading.Barrier(2)
    errors = []
    results = {}
    result_lock = threading.Lock()

    def run_client(client_name, ctrl_value):
        manager = get_pt_scaffold_auto_patch_manager().enable()
        try:
            model = _model(weight=1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            input_model = _input_model(model, ctrl_value=ctrl_value)

            manager.on_receive(input_model, task_name="train", train_task_name="train")
            model.load_state_dict(input_model.params)
            barrier.wait(timeout=5)

            _sgd_step(model, optimizer)
            output_model = FLModel(params=_state_dict(model))
            manager.on_send(output_model)

            with result_lock:
                results[client_name] = (
                    float(model.weight.item()),
                    AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta,
                )
        except Exception as e:
            with result_lock:
                errors.append(e)
        finally:
            manager.disable()

    threads = [
        threading.Thread(target=run_client, args=("client_a", 0.5)),
        threading.Thread(target=run_client, args=("client_b", 1.5)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    for thread in threads:
        assert not thread.is_alive()
    assert not errors
    assert results["client_a"][0] == pytest.approx(0.85)
    assert results["client_a"][1]
    assert results["client_b"][0] == pytest.approx(0.75)
    assert results["client_b"][1]


def test_send_fails_without_model_load(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    manager.on_receive(_input_model(model), task_name="train", train_task_name="train")

    with pytest.raises(RuntimeError, match="load_state_dict"):
        manager.on_send(FLModel(params=_state_dict(model)))


def test_send_fails_without_optimizer_step(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    input_model = _input_model(model)
    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)

    with pytest.raises(RuntimeError, match="optimizer.step"):
        manager.on_send(FLModel(params=_state_dict(model)))


def test_send_fails_when_manual_ctrl_diff_is_present(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_model = _input_model(model)
    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)
    _sgd_step(model, optimizer)

    output_model = FLModel(
        params=_state_dict(model),
        meta={AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"weight": torch.ones(1).numpy()}},
    )
    with pytest.raises(RuntimeError, match="already contains"):
        manager.on_send(output_model)


def test_optimizer_with_multiple_learning_rates_fails(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    input_model = _input_model(model)
    optimizer = torch.optim.SGD(
        [
            {"params": model[0].parameters(), "lr": 0.1},
            {"params": model[1].parameters(), "lr": 0.2},
        ]
    )

    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)
    loss = model(torch.tensor([[1.0]])).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="multiple different learning rates"):
        optimizer.step()


def test_learning_rate_scheduler_change_fails(clean_auto_patch_manager):
    manager = clean_auto_patch_manager.enable()
    model = _model(weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_model = _input_model(model)

    manager.on_receive(input_model, task_name="train", train_task_name="train")
    model.load_state_dict(input_model.params)
    _sgd_step(model, optimizer)

    optimizer.param_groups[0]["lr"] = 0.05
    with pytest.raises(RuntimeError, match="constant learning rate"):
        _sgd_step(model, optimizer)
