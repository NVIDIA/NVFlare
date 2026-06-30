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
from nvflare.app_opt.lightning.scaffold import _ScaffoldHandler, _StateDictSnapshot
from nvflare.client.config import ConfigKey


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
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


def _trainer(*optimizers):
    return SimpleNamespace(optimizers=list(optimizers))


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


def test_algorithm_handler_manager_does_not_load_scaffold_for_fedavg():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()

    with patch("nvflare.app_opt.lightning.algorithm._create_scaffold_handler") as create_handler:
        manager.start_round(_trainer(optimizer), module, FLModel(meta={}))
        manager.before_optimizer_step(optimizer)
        manager.after_train_batch(module)
        result = manager.finish_round(module)

    create_handler.assert_not_called()
    assert manager._handler is None
    assert result == _AlgorithmResult()


def test_algorithm_handler_manager_loads_scaffold_lazily_and_preserves_later_round_validation():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    manager = _AlgorithmHandlerManager()

    manager.start_round(_trainer(optimizer), module, _scaffold_model(module))
    handler = manager._handler
    manager.before_optimizer_step(optimizer)
    manager.after_train_batch(module)
    result = manager.finish_round(module)

    assert isinstance(handler, _ScaffoldHandler)
    assert result.num_steps == 1
    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in result.metadata
    with pytest.raises(RuntimeError, match="active in an earlier training round"):
        manager.start_round(_trainer(optimizer), module, FLModel(meta={}))


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


def test_scaffold_handler_skips_gradient_accumulation_batches_without_optimizer_step():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    handler.after_train_batch(module)
    assert handler._num_steps == 0

    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    assert handler._num_steps == 1


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


def test_scaffold_handler_uses_cpu_state_snapshot_without_deepcopying_module_each_round():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()

    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))
    initial_snapshot = _clone_state_dict(module)
    assert all(value.device.type == "cpu" for value in handler._model_global.state_dict().values())
    handler.before_optimizer_step(optimizer)
    handler.after_train_batch(module)
    handler.finish_round(module)

    module.non_copyable = threading.Lock()
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


def test_scaffold_handler_rejects_multiple_optimizers():
    module = SimpleNet()
    optimizer_1 = torch.optim.SGD([module.fc.weight], lr=0.1)
    optimizer_2 = torch.optim.SGD([module.fc.bias], lr=0.1)

    with pytest.raises(RuntimeError, match="requires exactly one optimizer"):
        _ScaffoldHandler().start_round(_trainer(optimizer_1, optimizer_2), module, _scaffold_model(module))


def test_scaffold_handler_rejects_unequal_parameter_group_learning_rates():
    module = SimpleNet()
    optimizer = torch.optim.SGD([{"params": module.fc.weight, "lr": 0.1}, {"params": module.fc.bias, "lr": 0.2}])
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    with pytest.raises(RuntimeError, match="all optimizer parameter groups"):
        handler.before_optimizer_step(optimizer)


def test_scaffold_handler_rejects_missing_control_keys():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    controls = _zero_controls(module)
    controls.pop("fc.bias")

    with pytest.raises(RuntimeError, match="must exactly match"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module, controls))


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

    with pytest.raises(RuntimeError, match="Invalid SCAFFOLD global controls"):
        _ScaffoldHandler().start_round(_trainer(optimizer), module, _scaffold_model(module, controls))


def test_scaffold_handler_rejects_round_without_optimizer_step():
    module = SimpleNet()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    handler = _ScaffoldHandler()
    handler.start_round(_trainer(optimizer), module, _scaffold_model(module))

    with pytest.raises(RuntimeError, match="at least one completed optimizer step"):
        handler.finish_round(module)


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
    callback._algorithm_handler.finish_round = MagicMock(
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
    callback._algorithm_handler.finish_round = MagicMock(
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


def test_train_end_fedavg_keeps_estimated_step_count_behavior():
    callback = _make_callback()
    callback._is_training = True
    callback._send_model = MagicMock()
    callback.reset_state = MagicMock()
    trainer = SimpleNamespace(estimated_stepping_batches=6)

    callback.on_train_end(trainer, SimpleNet())

    output_model = callback._send_model.call_args.args[0]
    assert output_model.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 6


def test_validation_before_fit_reuses_pending_training_model():
    callback = _make_callback()
    input_model = _scaffold_model(SimpleNet())
    callback._receive_and_update_model = MagicMock(return_value=input_model)
    callback._algorithm_handler.start_round = MagicMock()
    callback._is_training = True
    module = SimpleNet()
    trainer = SimpleNamespace()

    callback.on_validation_start(trainer, module)
    callback._algorithm_handler.start_round.assert_not_called()
    assert callback._pending_train_model is input_model

    callback.on_train_start(trainer, module)
    callback._receive_and_update_model.assert_called_once_with(trainer, module)
    callback._algorithm_handler.start_round.assert_called_once_with(
        trainer=trainer, pl_module=module, input_model=input_model
    )
    assert callback._pending_train_model is None


def test_real_lightning_fit_with_gradient_accumulation_returns_scaffold_controls():
    module = TinyLightningNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module)},
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
    assert callback._algorithm_handler._handler._helper.cnt == 2


def test_real_lightning_train_with_evaluation_reuses_scaffold_model_and_returns_metrics():
    module = TinyLightningNet()
    input_model = FLModel(
        params=_clone_state_dict(module),
        meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: _zero_controls(module)},
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
