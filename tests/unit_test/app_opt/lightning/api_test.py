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

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.lightning.algorithm import _AlgorithmHandlerManager, _AlgorithmResult
from nvflare.app_opt.lightning.api import FLCallback
from nvflare.app_opt.lightning.api import patch as lightning_patch
from nvflare.app_opt.lightning.callbacks import RestoreState
from nvflare.app_opt.lightning.fedprox import _FedProxHandler
from nvflare.app_opt.lightning.scaffold import _ScaffoldHandler, _StateDictSnapshot
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.client.config import ConfigKey


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.automatic_optimization = True


class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.automatic_optimization = True


class TinyLightningNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        return F.cross_entropy(self.fc(inputs), labels)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = F.cross_entropy(self.fc(inputs), labels)
        self.log("val_loss", loss, on_epoch=True, batch_size=inputs.shape[0])
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _make_callback(load_state_dict_strict: bool = False) -> FLCallback:
    with patch("nvflare.app_opt.lightning.api.init"):
        with patch("nvflare.app_opt.lightning.api.get_config", return_value={}):
            return FLCallback(rank=0, load_state_dict_strict=load_state_dict_strict)


def _clone_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _zero_controls(model: nn.Module) -> dict:
    return {key: torch.zeros_like(value) for key, value in model.state_dict().items()}


def _scaffold_model(model: nn.Module, controls=None) -> FLModel:
    if controls is None:
        controls = _zero_controls(model)
    return FLModel(meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: controls})


def _fedprox_model(mu=0.1) -> FLModel:
    return FLModel(meta={AlgorithmConstants.FEDPROX_MU: mu})


def _trainer(*optimizers, precision="32-true"):
    return SimpleNamespace(optimizers=list(optimizers), precision=precision)


def test_receive_and_update_model_raises_when_no_keys_match_and_suggests_prefix():
    callback = _make_callback()
    module = SimpleNet()
    params = {
        "model.fc.weight": torch.ones_like(module.state_dict()["fc.weight"]),
        "model.fc.bias": torch.zeros_like(module.state_dict()["fc.bias"]),
    }
    callback._receive_model = lambda trainer: FLModel(params=params)

    with pytest.raises(RuntimeError, match="stripping common prefix 'model\\.'"):
        callback._receive_and_update_model(SimpleNamespace(), module)


def test_receive_and_update_model_raises_on_shape_mismatch():
    callback = _make_callback()
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["fc.weight"] = torch.ones(3, 4)
    callback._receive_model = lambda trainer: FLModel(params=params)

    with pytest.raises(RuntimeError, match=r"fc.weight: expected \(2, 4\), got \(3, 4\)"):
        callback._receive_and_update_model(SimpleNamespace(), module)


def test_receive_and_update_model_warns_on_unexpected_keys_when_some_match(caplog):
    callback = _make_callback()
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["fc.weight"] = torch.full_like(params["fc.weight"], 3.0)
    params["fc.bias"] = torch.full_like(params["fc.bias"], -2.0)
    params["model.fc.weight"] = torch.ones_like(module.state_dict()["fc.weight"])
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        callback._receive_and_update_model(SimpleNamespace(), module)

    assert "Ignoring 1 unexpected model parameter" in caplog.text
    assert torch.equal(module.state_dict()["fc.weight"], params["fc.weight"])
    assert torch.equal(module.state_dict()["fc.bias"], params["fc.bias"])


def test_receive_and_update_model_rejects_unexpected_keys_in_strict_mode(caplog):
    callback = _make_callback(load_state_dict_strict=True)
    module = SimpleNet()
    params = _clone_state_dict(module)
    params["model.fc.weight"] = torch.ones_like(module.state_dict()["fc.weight"])
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="Rejecting 1 unexpected model parameter"):
            callback._receive_and_update_model(SimpleNamespace(), module)

    assert "Ignoring 1 unexpected model parameter" not in caplog.text


def test_receive_and_update_model_logs_missing_keys_for_partial_loads(caplog):
    callback = _make_callback(load_state_dict_strict=False)
    module = SimpleNet()
    params = {"fc.weight": torch.full_like(module.state_dict()["fc.weight"], 4.0)}
    callback._receive_model = lambda trainer: FLModel(params=params)

    with caplog.at_level(logging.WARNING):
        callback._receive_and_update_model(SimpleNamespace(), module)

    assert "There were missing keys when loading the global state_dict: ['fc.bias']" in caplog.text
    assert torch.equal(module.state_dict()["fc.weight"], params["fc.weight"])


def test_receive_model_broadcasts_scaffold_metadata_to_non_root_rank():
    callback = _make_callback()
    callback.rank = 1
    module = SimpleNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module)},
    )
    strategy = MagicMock()
    strategy.broadcast.side_effect = [input_model, True, False, False]

    with patch("nvflare.app_opt.lightning.api.receive") as receive:
        result = callback._receive_model(SimpleNamespace(strategy=strategy))

    receive.assert_not_called()
    assert result is input_model
    assert AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL in result.meta
    assert callback._is_training is True


def test_patch_is_idempotent():
    trainer = SimpleNamespace(callbacks=[], global_rank=0)

    with (
        patch("nvflare.app_opt.lightning.api.fobs.register"),
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
    ):
        lightning_patch(trainer)
        lightning_patch(trainer)

    assert len([cb for cb in trainer.callbacks if isinstance(cb, FLCallback)]) == 1
    assert len([cb for cb in trainer.callbacks if isinstance(cb, RestoreState)]) == 1


def test_algorithm_handler_manager_does_not_load_handlers_for_fedavg():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()

    with (
        patch("nvflare.app_opt.lightning.algorithm._create_scaffold_handler") as create_scaffold,
        patch("nvflare.app_opt.lightning.algorithm._create_fedprox_handler") as create_fedprox,
    ):
        manager.start_round(_trainer(optimizer), module, FLModel(meta={}))
        manager.before_optimizer_step(optimizer)
        manager.after_train_batch(module)
        result = manager.finish_round(module)

    create_scaffold.assert_not_called()
    create_fedprox.assert_not_called()
    assert manager._handlers == {}
    assert result == _AlgorithmResult()


def test_algorithm_handler_manager_loads_scaffold_lazily_and_preserves_later_round_validation():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()

    manager.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler = manager._handlers[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL]
    manager.before_optimizer_step(optimizer)
    manager.after_train_batch(module)
    result = manager.finish_round(module)

    assert isinstance(handler, _ScaffoldHandler)
    assert result.num_steps == 1
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in result.metadata
    with pytest.raises(RuntimeError, match="active in an earlier training round"):
        manager.start_round(_trainer(optimizer), module, FLModel(meta={}))


def test_algorithm_handler_manager_composes_scaffold_and_fedprox():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()
    input_model = FLModel(
        meta={
            AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module),
            AlgorithmConstants.FEDPROX_MU: 0.2,
        }
    )

    manager.start_round(_trainer(optimizer), module, input_model)
    manager.before_optimizer_step(optimizer)
    optimizer.step()
    manager.after_train_batch(module)
    result = manager.finish_round(module)

    assert list(manager._handlers) == [
        AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL,
        AlgorithmConstants.FEDPROX_MU,
    ]
    assert result.num_steps == 1
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in result.metadata


def test_algorithm_handler_manager_composes_scaffold_with_explicitly_disabled_fedprox():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()
    input_model = FLModel(
        meta={
            AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module),
            AlgorithmConstants.FEDPROX_MU: 0.0,
        }
    )

    manager.start_round(_trainer(optimizer), module, input_model)
    manager.before_optimizer_step(optimizer)
    optimizer.step()
    manager.after_train_batch(module)
    result = manager.finish_round(module)

    fedprox_handler = manager._handlers[AlgorithmConstants.FEDPROX_MU]
    assert fedprox_handler.active is False
    assert fedprox_handler._global_parameters is None
    assert result.num_steps == 1
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in result.metadata


class _FakeAlgorithmHandler:
    def __init__(self, num_steps, metadata=None, error=None):
        self.active = True
        self.num_steps = num_steps
        self.metadata = metadata or {}
        self.error = error
        self.finished = False

    def finish_round(self, pl_module):
        self.finished = True
        if self.error:
            raise self.error
        return self.metadata


def test_algorithm_handler_manager_rejects_inconsistent_step_counts():
    manager = _AlgorithmHandlerManager()
    manager._handlers = {
        "one": _FakeAlgorithmHandler(1, {}),
        "two": _FakeAlgorithmHandler(2, {}),
    }

    with pytest.raises(RuntimeError, match="inconsistent completed optimizer step counts"):
        manager.finish_round(SimpleNet())


def test_algorithm_handler_manager_rejects_metadata_collisions():
    manager = _AlgorithmHandlerManager()
    manager._handlers = {
        "one": _FakeAlgorithmHandler(1, {"reserved": 1}),
        "two": _FakeAlgorithmHandler(1, {"reserved": 2}),
    }

    with pytest.raises(RuntimeError, match="conflicting metadata keys"):
        manager.finish_round(SimpleNet())


def test_algorithm_handler_manager_logs_secondary_errors_and_finishes_every_handler(caplog):
    first = _FakeAlgorithmHandler(1, error=RuntimeError("first handler failed"))
    second = _FakeAlgorithmHandler(1, error=ValueError("second handler failed"))
    manager = _AlgorithmHandlerManager()
    manager._handlers = {"one": first, "two": second}

    with caplog.at_level(logging.ERROR, logger="nvflare.app_opt.lightning.algorithm"):
        with pytest.raises(RuntimeError, match="first handler failed"):
            manager.finish_round(SimpleNet())

    assert first.finished is True
    assert second.finished is True
    assert "second handler failed" in caplog.text


def test_fedprox_handler_matches_explicit_loss_with_momentum():
    torch.manual_seed(7)
    automatic_model = SimpleNet()
    explicit_model = SimpleNet()
    explicit_model.load_state_dict(automatic_model.state_dict())
    global_model = SimpleNet()
    global_model.load_state_dict(automatic_model.state_dict())
    automatic_optimizer = torch.optim.SGD(automatic_model.parameters(), lr=0.05, momentum=0.9)
    explicit_optimizer = torch.optim.SGD(explicit_model.parameters(), lr=0.05, momentum=0.9)
    handler = _FedProxHandler()
    handler.start_round(_trainer(automatic_optimizer), automatic_model, _fedprox_model(mu=0.3))
    criterion = PTFedProxLoss(mu=0.3)

    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    for _ in range(2):
        automatic_optimizer.zero_grad()
        F.mse_loss(automatic_model.fc(inputs), targets).backward()
        handler.before_optimizer_step(automatic_optimizer)
        automatic_optimizer.step()
        handler.after_train_batch(automatic_model)

        explicit_optimizer.zero_grad()
        explicit_loss = F.mse_loss(explicit_model.fc(inputs), targets) + criterion(explicit_model, global_model)
        explicit_loss.backward()
        explicit_optimizer.step()

    assert handler.num_steps == 2
    for automatic_parameter, explicit_parameter in zip(automatic_model.parameters(), explicit_model.parameters()):
        assert torch.allclose(automatic_parameter, explicit_parameter, atol=1e-7, rtol=1e-6)
    assert handler.finish_round(automatic_model) == {}
    assert handler._global_parameters is None


@pytest.mark.parametrize("invalid_mu", [-0.1, float("inf"), float("nan"), True, "0.1"])
def test_fedprox_handler_rejects_invalid_metadata(invalid_mu):
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="finite non-negative number"):
        _FedProxHandler().start_round(_trainer(optimizer), module, _fedprox_model(invalid_mu))


def test_fedprox_handler_allows_positive_mu_change_but_rejects_later_missing_metadata():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()

    for mu in (0.1, 0.4):
        handler.start_round(_trainer(optimizer), module, _fedprox_model(mu))
        handler.before_optimizer_step(optimizer)
        optimizer.step()
        handler.after_train_batch(module)
        handler.finish_round(module)

    with pytest.raises(RuntimeError, match="metadata was received in an earlier training round"):
        handler.start_round(_trainer(optimizer), module, FLModel(meta={}))


def test_fedprox_handler_explicit_zero_disables_without_optimizer_validation_or_step():
    module = SimpleNet()
    module.automatic_optimization = False
    handler = _FedProxHandler()

    handler.start_round(_trainer(), module, _fedprox_model(0.0))
    module.fc.weight.grad = torch.ones_like(module.fc.weight)
    original_gradient = module.fc.weight.grad.clone()
    handler.before_optimizer_step(None)
    handler.after_train_batch(module)

    assert handler.active is False
    assert handler.num_steps == 0
    assert handler._global_parameters is None
    assert torch.equal(module.fc.weight.grad, original_gradient)
    assert handler.finish_round(module) == {}


def test_fedprox_handler_supports_zero_positive_zero_positive_schedule():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()

    handler.start_round(_trainer(), module, _fedprox_model(0.0))
    assert handler.active is False
    assert handler._global_parameters is None
    assert handler.finish_round(module) == {}

    handler.start_round(_trainer(optimizer), module, _fedprox_model(0.1))
    first_snapshot = {name: value.clone() for name, value in handler._global_parameters.items()}
    handler.before_optimizer_step(optimizer)
    optimizer.step()
    handler.after_train_batch(module)
    handler.finish_round(module)

    handler.start_round(_trainer(), module, _fedprox_model(0.0))
    assert handler.active is False
    assert handler._global_parameters is None
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(1.0)
    assert handler.finish_round(module) == {}

    handler.start_round(_trainer(optimizer), module, _fedprox_model(0.4))
    assert handler.active is True
    assert handler._mu == 0.4
    assert any(
        not torch.equal(handler._global_parameters[name], first_snapshot[name]) for name in handler._global_parameters
    )
    handler.before_optimizer_step(optimizer)
    optimizer.step()
    handler.after_train_batch(module)
    handler.finish_round(module)


def test_fedprox_handler_rejects_missing_metadata_after_explicit_zero():
    handler = _FedProxHandler()
    handler.start_round(_trainer(), SimpleNet(), _fedprox_model(0.0))

    with pytest.raises(RuntimeError, match="explicit 0.0"):
        handler.start_round(_trainer(), SimpleNet(), FLModel(meta={}))


def test_fedprox_handler_snapshots_only_optimizer_owned_trainable_parameters():
    module = BatchNormNet()
    module.fc.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()
    handler.start_round(_trainer(optimizer), module, _fedprox_model())

    snapshot_keys = set(handler._global_parameters)
    expected_keys = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert snapshot_keys == expected_keys
    assert snapshot_keys.isdisjoint(dict(module.named_buffers()))
    assert "fc.bias" not in snapshot_keys

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(1.0)
    handler.before_optimizer_step(optimizer)
    assert module.fc.bias.grad is None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert torch.allclose(parameter.grad, torch.full_like(parameter, 0.1))
    optimizer.step()
    handler.after_train_batch(module)
    handler.finish_round(module)


def test_fedprox_handler_rejects_manual_optimization():
    module = SimpleNet()
    module.automatic_optimization = False
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="requires automatic optimization"):
        _FedProxHandler().start_round(_trainer(optimizer), module, _fedprox_model())


@pytest.mark.parametrize("precision", ["16-mixed", "64-true"])
def test_fedprox_handler_rejects_unsupported_precision(precision):
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="requires trainer.precision"):
        _FedProxHandler().start_round(_trainer(optimizer, precision=precision), module, _fedprox_model())


def test_fedprox_handler_rejects_scaler_backed_precision():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    trainer = _trainer(optimizer)
    trainer.precision_plugin = SimpleNamespace(scaler=object())

    with pytest.raises(RuntimeError, match="scaler-backed precision"):
        _FedProxHandler().start_round(trainer, module, _fedprox_model())


def test_fedprox_handler_rejects_multiple_optimizers():
    module = SimpleNet()
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.1) for _ in range(2)]

    with pytest.raises(RuntimeError, match="exactly one optimizer"):
        _FedProxHandler().start_round(_trainer(*optimizers), module, _fedprox_model())


@pytest.mark.parametrize("optimizer_class", [torch.optim.LBFGS, torch.optim.SparseAdam])
def test_fedprox_handler_rejects_unsupported_optimizer(optimizer_class):
    module = SimpleNet()
    optimizer = optimizer_class(module.parameters())

    with pytest.raises(RuntimeError, match="does not support"):
        _FedProxHandler().start_round(_trainer(optimizer), module, _fedprox_model())


def test_fedprox_handler_rejects_optimizer_parameters_outside_module():
    module = SimpleNet()
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)

    with pytest.raises(RuntimeError, match="not owned by the LightningModule"):
        _FedProxHandler().start_round(_trainer(optimizer), module, _fedprox_model())


def test_fedprox_handler_rejects_sparse_gradients_before_mutation():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()
    handler.start_round(_trainer(optimizer), module, _fedprox_model())
    module.fc.weight.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), size=module.fc.weight.shape
    )

    with pytest.raises(RuntimeError, match="requires dense gradients"):
        handler.before_optimizer_step(optimizer)


def test_fedprox_handler_rejects_mid_round_trainability_change():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()
    handler.start_round(_trainer(optimizer), module, _fedprox_model())
    module.fc.bias.requires_grad_(False)

    with pytest.raises(RuntimeError, match="trainability change"):
        handler.before_optimizer_step(optimizer)


def test_fedprox_handler_rejects_round_without_completed_step_and_cleans_snapshot():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()
    handler.start_round(_trainer(optimizer), module, _fedprox_model())

    with pytest.raises(RuntimeError, match="at least one completed optimizer step"):
        handler.finish_round(module)
    assert handler._global_parameters is None


def test_fedprox_handler_explains_interrupted_optimizer_hook_sequence():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _FedProxHandler()
    handler.start_round(_trainer(optimizer), module, _fedprox_model())
    handler.before_optimizer_step(optimizer)

    with pytest.raises(RuntimeError, match="did not call on_train_batch_end"):
        handler.finish_round(module)


def test_scaffold_handler_is_noop_without_scaffold_metadata():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer), module, FLModel(meta={}))

    assert handler.active is False
    assert handler.finish_round(module) == {}


def test_scaffold_handler_applies_post_step_update_and_returns_control_diff():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = {key: torch.ones_like(value) for key, value in module.state_dict().items()}
    handler = _ScaffoldHandler()
    initial_state = _clone_state_dict(module)

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module, controls))
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    result = handler.finish_round(module)

    for key, value in module.state_dict().items():
        assert torch.allclose(value, initial_state[key] - 0.1)
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in result
    assert set(result[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]) == set(module.state_dict())


def test_scaffold_handler_corrects_parameters_without_modifying_batch_norm_buffers():
    module = BatchNormNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = {key: torch.ones_like(value) for key, value in module.state_dict().items()}
    handler = _ScaffoldHandler()
    initial_state = _clone_state_dict(module)

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module, controls))
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    result = handler.finish_round(module)

    parameter_keys = {key for key, parameter in module.named_parameters() if parameter.requires_grad}
    buffer_keys = set(module.state_dict()) - parameter_keys
    for key in parameter_keys:
        assert torch.allclose(module.state_dict()[key], initial_state[key] - 0.1)
    for key in buffer_keys:
        assert torch.equal(module.state_dict()[key], initial_state[key])
    assert set(result[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]) == parameter_keys


def test_scaffold_handler_uses_average_step_lr_for_terms_update():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    optimizer.param_groups[0]["lr"] = 0.2
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)

    with patch.object(handler._helper, "terms_update", wraps=handler._helper.terms_update) as terms_update:
        handler.finish_round(module)

    assert terms_update.call_args.kwargs["curr_lr"] == pytest.approx(0.15)


def test_scaffold_handler_allows_zero_lr_warmup_when_round_has_positive_total_exposure():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    optimizer.param_groups[0]["lr"] = 0.2
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)

    with patch.object(handler._helper, "terms_update", wraps=handler._helper.terms_update) as terms_update:
        handler.finish_round(module)

    assert handler._helper.cnt == 2
    assert terms_update.call_args.kwargs["curr_lr"] == pytest.approx(0.1)


def test_scaffold_handler_rejects_round_with_zero_total_lr_exposure():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)

    with pytest.raises(RuntimeError, match="positive total learning-rate exposure"):
        handler.finish_round(module)


def test_scaffold_handler_skips_gradient_accumulation_batches_without_optimizer_step():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    handler.after_train_batch(module)
    assert handler.num_steps == 0

    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    assert handler.num_steps == 1


def test_scaffold_handler_preserves_local_controls_between_rounds_and_rejects_missing_controls():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    helper = handler._helper
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    handler.finish_round(module)

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    assert handler._helper is helper

    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    handler.finish_round(module)
    with pytest.raises(RuntimeError, match="active in an earlier training round"):
        handler.start_round(_trainer(optimizer), module, FLModel(meta={}))


def test_scaffold_handler_uses_cpu_controls_and_snapshot_without_deepcopying_module():
    module = SimpleNet()
    module.non_copyable = threading.Lock()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    initial_snapshot = _clone_state_dict(module)
    assert all(value.device.type == "cpu" for value in handler._model_global.state_dict().values())
    assert all(value.device.type == "cpu" for value in handler._helper.c_global.state_dict().values())
    assert all(value.device.type == "cpu" for value in handler._helper.c_local.state_dict().values())
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    assert set(handler._helper._control_correction) == set(dict(module.named_parameters()))
    handler.finish_round(module)
    assert handler._helper._control_correction == {}

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    with torch.no_grad():
        module.fc.weight.add_(1)

    for key, value in handler._model_global.state_dict().items():
        assert torch.equal(value, initial_snapshot[key])


def test_state_dict_snapshot_is_independent_and_cpu_backed():
    module = SimpleNet()
    expected = _clone_state_dict(module)
    snapshot = _StateDictSnapshot(module.state_dict())

    with torch.no_grad():
        module.fc.weight.add_(1)

    for key, value in snapshot.state_dict().items():
        assert value.device.type == "cpu"
        assert torch.equal(value, expected[key])


def test_scaffold_handler_rejects_manual_optimization():
    module = SimpleNet()
    module.automatic_optimization = False
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="requires automatic optimization"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module))


def test_scaffold_handler_rejects_gradient_scaler_mixed_precision():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    trainer = _trainer(optimizer)
    trainer.scaler = MagicMock()

    with pytest.raises(RuntimeError, match="does not support mixed precision that uses a gradient scaler"):
        _ScaffoldHandler().start_round(trainer, module, _scaffold_model(module))


def test_scaffold_handler_rejects_unsupported_precision_without_scaler_attributes():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="trainer.precision"):
        _ScaffoldHandler().start_round(_trainer(optimizer, precision="16-mixed"), module, _scaffold_model(module))


def test_scaffold_handler_accepts_bf16_mixed_precision_without_scaler():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer, precision="bf16-mixed"), module, _scaffold_model(module))

    assert handler.active is True


def test_scaffold_handler_rejects_multiple_optimizers():
    module = SimpleNet()
    optimizer_1 = torch.optim.SGD([module.fc.weight], lr=0.1)
    optimizer_2 = torch.optim.SGD([module.fc.bias], lr=0.1)

    with pytest.raises(RuntimeError, match="requires exactly one optimizer"):
        _ScaffoldHandler().start_round(_trainer(optimizer_1, optimizer_2), module, _scaffold_model(module))


def test_scaffold_handler_rejects_unequal_parameter_group_learning_rates():
    module = SimpleNet()
    optimizer = torch.optim.SGD([{"params": module.fc.weight, "lr": 0.1}, {"params": module.fc.bias, "lr": 0.2}])
    with pytest.raises(RuntimeError, match="all optimizer parameter groups"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module))


def test_scaffold_handler_rejects_scheduler_induced_unequal_parameter_group_learning_rates():
    module = SimpleNet()
    optimizer = torch.optim.SGD([{"params": module.fc.weight}, {"params": module.fc.bias}], lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    optimizer.param_groups[1]["lr"] = 0.2

    with pytest.raises(RuntimeError, match="all optimizer parameter groups"):
        handler.before_optimizer_step(optimizer)


@pytest.mark.parametrize("invalid_lr", [-0.1, float("inf"), float("nan")])
def test_scaffold_handler_rejects_negative_or_non_finite_learning_rate(invalid_lr):
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    optimizer.param_groups[0]["lr"] = invalid_lr

    with pytest.raises(RuntimeError, match="finite non-negative learning rate"):
        handler.before_optimizer_step(optimizer)


def test_scaffold_handler_zero_fills_missing_parameter_controls_and_ignores_global_only_keys():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = _zero_controls(module)
    controls.pop("fc.bias")
    controls["server_only.buffer"] = torch.ones(1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module, controls))

    assert set(handler._c_global_para) == set(dict(module.named_parameters()))
    assert torch.count_nonzero(handler._c_global_para["fc.bias"]) == 0


def test_scaffold_handler_rejects_present_but_incomplete_control_metadata():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    input_model = FLModel(meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: None})

    with pytest.raises(RuntimeError, match="requires non-empty mapping metadata"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, input_model)


def test_scaffold_handler_rejects_control_shape_mismatch():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = _zero_controls(module)
    controls["fc.weight"] = torch.zeros(3, 4)

    with pytest.raises(RuntimeError, match="expected shape"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module, controls))


def test_scaffold_handler_rejects_controls_without_matching_parameters():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = {"server_only.weight": torch.zeros_like(module.fc.weight)}

    with pytest.raises(RuntimeError, match="do not match any named parameters"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module, controls))


def test_scaffold_handler_rejects_non_tensor_control_value():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = _zero_controls(module)
    controls["fc.weight"] = object()

    with pytest.raises(RuntimeError, match="Failed to convert SCAFFOLD global control 'fc.weight'"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module, controls))


def test_scaffold_handler_resets_newly_trainable_local_control_between_rounds():
    module = SimpleNet()
    module.fc.bias.requires_grad = False
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    handler.finish_round(module)

    handler._helper.c_local._values["fc.bias"].fill_(3)
    module.fc.bias.requires_grad = True
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    assert torch.count_nonzero(handler._c_local_para["fc.bias"]) == 0


def test_scaffold_handler_rejects_trainability_change_during_round():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    module.fc.bias.requires_grad = False
    handler.before_optimizer_step(optimizer)

    with pytest.raises(RuntimeError, match="changing requires_grad during a training round"):
        handler.after_train_batch(module)


def test_scaffold_handler_rejects_round_without_optimizer_step():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    with pytest.raises(RuntimeError, match="at least one completed optimizer step"):
        handler.finish_round(module)
    assert handler.active is False
    assert handler._model_global is None


def test_scaffold_handler_cleans_round_state_when_terms_update_fails():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)

    with patch.object(handler._helper, "terms_update", side_effect=RuntimeError("terms update failed")):
        with pytest.raises(RuntimeError, match="terms update failed"):
            handler.finish_round(module)

    assert handler.active is False
    assert handler._model_global is None


def test_scaffold_handler_explains_interrupted_optimizer_hook_sequence():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler.before_optimizer_step(optimizer)

    with pytest.raises(RuntimeError, match="did not call on_train_batch_end"):
        handler.finish_round(module)


def test_train_end_uses_scaffold_completed_steps_and_preserves_user_metadata():
    callback = _make_callback()
    callback._is_training = True
    callback._algorithm_handler_manager.finish_round = MagicMock(
        return_value=_AlgorithmResult(metadata={AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"fc": 1}}, num_steps=3)
    )
    callback._send_model = MagicMock()
    callback.reset_state = MagicMock()
    module = SimpleNet()
    module.__fl_meta__ = {"custom": "value"}
    trainer = SimpleNamespace(estimated_stepping_batches=6)

    callback.on_train_end(trainer, module)

    output_model = callback._send_model.call_args.args[0]
    assert output_model.meta["custom"] == "value"
    assert output_model.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 3
    assert output_model.meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF] == {"fc": 1}
    assert module.__fl_meta__ == {"custom": "value"}


def test_train_end_preserves_explicit_user_step_count_for_scaffold():
    callback = _make_callback()
    callback._is_training = True
    callback._algorithm_handler_manager.finish_round = MagicMock(
        return_value=_AlgorithmResult(metadata={AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"fc": 1}}, num_steps=3)
    )
    callback._send_model = MagicMock()
    callback.reset_state = MagicMock()
    module = SimpleNet()
    module.__fl_meta__ = {MetaKey.NUM_STEPS_CURRENT_ROUND: 5}
    trainer = SimpleNamespace(estimated_stepping_batches=6)

    callback.on_train_end(trainer, module)

    output_model = callback._send_model.call_args.args[0]
    assert output_model.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 5


def test_train_end_rejects_user_metadata_that_conflicts_with_automatic_algorithm_metadata():
    callback = _make_callback()
    callback._is_training = True
    callback._algorithm_handler_manager.finish_round = MagicMock(
        return_value=_AlgorithmResult(metadata={AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"fc": 1}}, num_steps=3)
    )
    module = SimpleNet()
    module.__fl_meta__ = {AlgorithmConstants.SCAFFOLD_CTRL_DIFF: {"manual": 1}}

    with pytest.raises(RuntimeError, match="conflicts with user-provided"):
        callback.on_train_end(SimpleNamespace(estimated_stepping_batches=3), module)


def test_train_end_fedavg_reports_completed_steps_for_each_round():
    callback = _make_callback()
    callback._is_training = True
    callback._send_model = MagicMock()
    callback.reset_state = MagicMock()
    trainer = SimpleNamespace(global_step=5)
    callback._round_start_global_step = 2

    callback.on_train_end(trainer, SimpleNet())
    callback._round_start_global_step = 5
    trainer.global_step = 9
    callback.on_train_end(trainer, SimpleNet())

    assert callback._send_model.call_args_list[0].args[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 3
    assert callback._send_model.call_args_list[1].args[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 4


def test_validation_before_fit_reuses_pending_training_model():
    callback = _make_callback()
    input_model = _scaffold_model(SimpleNet())
    callback._receive_and_update_model = MagicMock(return_value=input_model)
    callback._update_model = MagicMock()
    callback._algorithm_handler_manager.start_round = MagicMock()
    callback._is_training = True
    module = SimpleNet()
    trainer = SimpleNamespace(global_step=0)

    callback.on_validation_start(trainer, module)
    callback._algorithm_handler_manager.start_round.assert_not_called()
    assert callback._pending_train_model is input_model

    callback.on_train_start(trainer, module)
    callback._receive_and_update_model.assert_called_once_with(trainer, module)
    callback._update_model.assert_called_once_with(module, input_model)
    callback._algorithm_handler_manager.start_round.assert_called_once_with(
        trainer=trainer, pl_module=module, input_model=input_model
    )
    assert callback._pending_train_model is None
    assert callback._training_round_started is True


def test_validation_before_fit_reapplies_pending_model_to_a_different_module():
    callback = _make_callback()
    input_model = _scaffold_model(SimpleNet())
    callback._receive_and_update_model = MagicMock(return_value=input_model)
    callback._update_model = MagicMock()
    callback._algorithm_handler_manager.start_round = MagicMock()
    callback._is_training = True
    validation_module = SimpleNet()
    training_module = SimpleNet()
    trainer = SimpleNamespace(global_step=0)

    callback.on_validation_start(trainer, validation_module)
    callback.on_train_start(trainer, training_module)

    callback._receive_and_update_model.assert_called_once_with(trainer, validation_module)
    callback._update_model.assert_called_once_with(training_module, input_model)
    callback._algorithm_handler_manager.start_round.assert_called_once_with(
        trainer=trainer, pl_module=training_module, input_model=input_model
    )


def test_validation_before_fit_reapplies_pending_model_to_same_module_after_mutation():
    callback = _make_callback()
    module = SimpleNet()
    global_params = {key: torch.full_like(value, 2) for key, value in module.state_dict().items()}
    input_model = FLModel(params=global_params)
    callback._receive_and_update_model = MagicMock(return_value=input_model)
    callback._is_training = True
    trainer = SimpleNamespace(global_step=0)

    callback.on_validation_start(trainer, module)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    callback.on_train_start(trainer, module)

    for key, value in module.state_dict().items():
        assert torch.equal(value, global_params[key])


def test_mid_fit_validation_collects_metrics_without_receiving_or_reloading_model():
    callback = _make_callback()
    callback._training_round_started = True
    callback._receive_and_update_model = MagicMock()
    trainer = SimpleNamespace(callback_metrics={"val_loss": torch.tensor(0.5)})

    callback.on_validation_start(trainer, SimpleNet())
    callback.on_validation_end(trainer, SimpleNet())

    callback._receive_and_update_model.assert_not_called()
    assert callback.metrics == {"val_loss": 0.5}


def test_real_lightning_fit_with_fedprox_and_gradient_accumulation():
    module = TinyLightningNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={AlgorithmConstants.FEDPROX_MU: 0.2},
    )
    train_loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=1)

    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
        patch("nvflare.app_opt.lightning.api.receive", return_value=input_model),
        patch("nvflare.app_opt.lightning.api.is_train", return_value=True),
        patch("nvflare.app_opt.lightning.api.is_evaluate", return_value=False),
        patch("nvflare.app_opt.lightning.api.is_submit_model", return_value=False),
        patch("nvflare.app_opt.lightning.api.send") as send,
        patch("nvflare.app_opt.lightning.api.clear"),
    ):
        callback = FLCallback(rank=0)
        trainer = pl.Trainer(
            max_epochs=1,
            accumulate_grad_batches=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=train_loader)

    output_model = send.call_args.args[0]
    assert output_model.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 2
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF not in output_model.meta
    handler = callback._algorithm_handler_manager._handlers[AlgorithmConstants.FEDPROX_MU]
    assert handler.active is False
    assert handler._global_parameters is None


def test_real_lightning_fit_with_scaffold_fedprox_and_gradient_accumulation_returns_controls():
    module = TinyLightningNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={
            AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module),
            AlgorithmConstants.FEDPROX_MU: 0.2,
        },
    )
    train_loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=1, shuffle=False)

    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
        patch("nvflare.app_opt.lightning.api.receive", return_value=input_model),
        patch("nvflare.app_opt.lightning.api.is_train", return_value=True),
        patch("nvflare.app_opt.lightning.api.is_evaluate", return_value=False),
        patch("nvflare.app_opt.lightning.api.is_submit_model", return_value=False),
        patch("nvflare.app_opt.lightning.api.send") as send,
        patch("nvflare.app_opt.lightning.api.clear"),
    ):
        callback = FLCallback(rank=0)
        trainer = pl.Trainer(
            max_epochs=1,
            accumulate_grad_batches=2,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=train_loader)

    output_model = send.call_args.args[0]
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta
    assert output_model.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 2
    scaffold_handler = callback._algorithm_handler_manager._handlers[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL]
    fedprox_handler = callback._algorithm_handler_manager._handlers[AlgorithmConstants.FEDPROX_MU]
    assert scaffold_handler._helper.cnt == 2
    assert fedprox_handler.active is False


def test_real_lightning_train_with_evaluation_reuses_scaffold_model_and_returns_metrics():
    module = TinyLightningNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={
            AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module),
            AlgorithmConstants.FEDPROX_MU: 0.2,
        },
    )
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=1, shuffle=False)

    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch(
            "nvflare.app_opt.lightning.api.get_config",
            return_value={ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}},
        ),
        patch("nvflare.app_opt.lightning.api.receive", return_value=input_model) as receive,
        patch("nvflare.app_opt.lightning.api.is_train", return_value=True),
        patch("nvflare.app_opt.lightning.api.is_evaluate", return_value=False),
        patch("nvflare.app_opt.lightning.api.is_submit_model", return_value=False),
        patch("nvflare.app_opt.lightning.api.send") as send,
        patch("nvflare.app_opt.lightning.api.clear"),
    ):
        callback = FLCallback(rank=0)
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=1,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    output_model = send.call_args.args[0]
    assert receive.call_count == 1
    assert output_model.metrics["val_loss"] >= 0.0
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in output_model.meta
    assert callback._pending_train_model is None


def test_real_lightning_fedavg_reports_per_round_steps_across_two_fits():
    module = TinyLightningNet()
    input_models = [
        FLModel(params=_clone_state_dict(module), current_round=round_idx, meta={}) for round_idx in range(2)
    ]
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=1, shuffle=False)

    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
        patch("nvflare.app_opt.lightning.api.receive", side_effect=input_models),
        patch("nvflare.app_opt.lightning.api.is_train", return_value=True),
        patch("nvflare.app_opt.lightning.api.is_evaluate", return_value=False),
        patch("nvflare.app_opt.lightning.api.is_submit_model", return_value=False),
        patch("nvflare.app_opt.lightning.api.send") as send,
        patch("nvflare.app_opt.lightning.api.clear"),
    ):
        callback = FLCallback(rank=0)
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=loader)
        trainer.fit(module, train_dataloaders=loader)

    assert [call.args[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] for call in send.call_args_list] == [4, 4]


def test_real_lightning_fit_only_reuses_one_received_model_per_round_with_mid_fit_validation():
    module = TinyLightningNet()
    global_params = {key: torch.zeros_like(value) for key, value in module.state_dict().items()}
    input_models = [
        FLModel(
            params={key: value.clone() for key, value in global_params.items()},
            current_round=round_idx,
            meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module)},
        )
        for round_idx in range(2)
    ]
    loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=1, shuffle=False)

    with (
        patch("nvflare.app_opt.lightning.api.init"),
        patch("nvflare.app_opt.lightning.api.get_config", return_value={}),
        patch("nvflare.app_opt.lightning.api.receive", side_effect=input_models) as receive,
        patch("nvflare.app_opt.lightning.api.is_train", return_value=True),
        patch("nvflare.app_opt.lightning.api.is_evaluate", return_value=False),
        patch("nvflare.app_opt.lightning.api.is_submit_model", return_value=False),
        patch("nvflare.app_opt.lightning.api.send") as send,
        patch("nvflare.app_opt.lightning.api.clear"),
    ):
        callback = FLCallback(rank=0)
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert receive.call_count == 2
    assert send.call_count == 2
    second_output = send.call_args_list[1].args[0]
    assert any(not torch.equal(second_output.params[key], global_params[key]) for key in global_params)
    assert set(second_output.meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]) == set(dict(module.named_parameters()))
    assert callback._training_round_started is False
