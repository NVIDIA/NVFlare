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
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey  # noqa: E402

from ._helpers import (  # noqa: E402
    ClientAPIMock,
    call_hf_callback,
    import_hf_module,
    make_fake_trainer_class,
    make_training_args,
    patch_client_api_aliases,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def bump_after_train(self):
        with torch.no_grad():
            self.fc.weight.add_(1.0)
            self.fc.bias.add_(1.0)


class TinyBF16Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    def bump_after_train(self):
        with torch.no_grad():
            self.weight.add_(torch.ones_like(self.weight))


def _fresh_api(monkeypatch, incoming_model, task="train", train_with_eval=False):
    client_api_mock = ClientAPIMock(incoming_model=incoming_model, task=task, train_with_eval=train_with_eval)
    patch_client_api_aliases(monkeypatch, client_api_mock)
    hf_api = import_hf_module(monkeypatch, "nvflare.app_opt.hf.api")
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    transformers = pytest.importorskip("transformers")
    trainer_cls = make_fake_trainer_class(transformers)
    return hf_api, trainer_cls, client_api_mock


def _make_trainer(trainer_cls, tmp_path):
    model = TinyModel()
    args = make_training_args(tmp_path, per_device_train_batch_size=4, gradient_accumulation_steps=2)
    return trainer_cls(model, args)


def _model_params(model, value):
    return {name: torch.full_like(param, value) for name, param in model.state_dict().items()}


class _RecordingDist:
    def __init__(self, rank=0, world_size=2, incoming_payload=None, all_gather_payload=None):
        self.rank = rank
        self.world_size = world_size
        self.incoming_payload = incoming_payload
        self.incoming_payloads = list(incoming_payload) if isinstance(incoming_payload, list) else None
        self.all_gather_payload = all_gather_payload
        self.all_gather_calls = []
        self.broadcast_payloads = []
        self.barrier_calls = 0

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def broadcast_object_list(self, payload, src=0):
        if self.rank == src:
            self.broadcast_payloads.append(dict(payload[0] or {}))
        else:
            if self.incoming_payloads is not None:
                payload[0] = self.incoming_payloads.pop(0)
            else:
                payload[0] = self.incoming_payload

    def all_gather_object(self, gathered, obj):
        self.all_gather_calls.append(dict(obj or {}))
        if self.all_gather_payload is not None:
            for idx, value in enumerate(self.all_gather_payload):
                gathered[idx] = value
            return
        for idx in range(len(gathered)):
            gathered[idx] = {
                "ok": True,
                "operation": obj.get("operation") if isinstance(obj, dict) else None,
                "rank": idx,
                "error": None,
            }

    def barrier(self):
        self.barrier_calls += 1


def _rank_zero_failure(operation: str, error: str):
    return {"ok": False, "operation": operation, "error": error}


def _task_payload(task_kind, call_name, current_round=1, total_rounds=2):
    return {
        "task_kind": task_kind,
        "call_name": call_name,
        "fl_model": None,
        "params": {},
        "current_round": current_round,
        "total_rounds": total_rounds,
    }


def test_train_task_receives_once_captures_pre_train_eval_and_sends_after_train(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, train_with_eval=True)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    metrics = trainer.evaluate()
    trainer.train()

    assert metrics == {"eval_loss": 0.25}
    assert client_api_mock.events.index("receive") < client_api_mock.events.index("send")
    assert client_api_mock.receive_calls == 1
    assert len(client_api_mock.sent_models) == 1
    assert client_api_mock.sent_models[0].metrics == {"eval_loss": 0.25}
    assert client_api_mock.sent_models[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 8


def test_restore_state_false_uses_fresh_trainer_state_each_round(monkeypatch, tmp_path):
    initial_model = TinyModel()
    first_round = FLModel(params=_model_params(initial_model, 5.0), current_round=0, total_rounds=2)
    second_round = FLModel(params=_model_params(initial_model, 7.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, first_round)
    client_api_mock.incoming_models.append(second_round)
    trainer = _make_trainer(trainer_cls, tmp_path)
    trainer_state_cls = type(trainer.state)
    reset_observations = []

    def make_trainer_stateful():
        trainer.optimizer = object()
        trainer.lr_scheduler = object()
        trainer._created_lr_scheduler = True
        trainer.control.should_training_stop = True

    def train_resetting_state(*args, **kwargs):
        trainer.train_call_count += 1
        trainer.last_train_args = args
        trainer.last_train_kwargs = dict(kwargs)
        reset_observations.append(
            (
                trainer.optimizer,
                trainer.lr_scheduler,
                getattr(trainer, "_created_lr_scheduler", None),
                trainer.control.should_training_stop,
            )
        )
        if trainer.train_call_count > 1:
            trainer.state = trainer_state_cls()
        for callback in list(trainer.callbacks):
            call_hf_callback(callback, "on_train_begin", trainer)
        trainer.state.global_step += 1
        for callback in list(trainer.callbacks):
            call_hf_callback(callback, "on_train_end", trainer)
        return SimpleNamespace(global_step=trainer.state.global_step)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, train_resetting_state)

    make_trainer_stateful()
    assert hf_api.hf_is_running()
    trainer.train()
    make_trainer_stateful()
    assert hf_api.hf_is_running()
    trainer.train()

    assert reset_observations == [(None, None, False, False), (None, None, False, False)]
    assert len(client_api_mock.sent_models) == 2
    assert client_api_mock.sent_models[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 8
    assert client_api_mock.sent_models[1].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 8
    assert trainer._nvflare_hf_task_state.metric_step_offset == 2


def test_train_task_uses_token_delta_for_weight_only_when_enabled(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = trainer_cls(
        TinyModel(),
        make_training_args(
            tmp_path,
            include_num_input_tokens_seen=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
        ),
    )

    def train_with_tokens(*args, **kwargs):
        trainer.train_call_count += 1
        trainer.last_train_args = args
        trainer.last_train_kwargs = dict(kwargs)
        for callback in list(trainer.callbacks):
            call_hf_callback(callback, "on_train_begin", trainer)
        trainer.state.num_input_tokens_seen = 37
        trainer.state.global_step += 1
        for callback in list(trainer.callbacks):
            call_hf_callback(callback, "on_train_end", trainer)
        return SimpleNamespace(global_step=trainer.state.global_step)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, train_with_tokens)

    trainer.train()

    assert client_api_mock.sent_models[0].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 37


def test_train_task_casts_bf16_tensors_when_server_expects_numpy(monkeypatch, tmp_path):
    from nvflare.client.config import ExchangeFormat

    model = TinyBF16Model()
    incoming_model = FLModel(params=_model_params(model, 5.0), current_round=1, total_rounds=3)
    client_api_mock = ClientAPIMock(
        incoming_model=incoming_model,
        exchange_format=ExchangeFormat.PYTORCH,
        server_expected_format=ExchangeFormat.NUMPY,
    )
    patch_client_api_aliases(monkeypatch, client_api_mock)
    hf_api = import_hf_module(monkeypatch, "nvflare.app_opt.hf.api")
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    transformers = pytest.importorskip("transformers")
    trainer_cls = make_fake_trainer_class(transformers)
    trainer = trainer_cls(model, make_training_args(tmp_path))

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer.train()

    sent_param = client_api_mock.sent_models[0].params["weight"]
    assert torch.is_tensor(sent_param)
    assert sent_param.dtype == torch.float32


def test_train_with_evaluation_fails_fast_when_pre_train_evaluate_was_not_called(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model, train_with_eval=True)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="missing.*metrics|remember to call evaluate"):
        trainer.train()


def test_second_pre_train_evaluate_does_not_overwrite_first_metrics(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, train_with_eval=True)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer.evaluate_metrics = {"eval_loss": 0.25}
    trainer.evaluate()
    trainer.evaluate_metrics = {"eval_loss": 0.10}
    trainer.evaluate()
    trainer.train()

    assert client_api_mock.sent_models[0].metrics == {"eval_loss": 0.25}


def test_evaluate_task_sends_metrics_only_and_subsequent_train_is_noop(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=1)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, task="evaluate")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    metrics = trainer.evaluate()
    train_result = trainer.train()

    assert metrics == {"eval_loss": 0.25}
    assert train_result is None
    assert trainer.train_call_count == 0
    assert len(client_api_mock.sent_models) == 1
    assert client_api_mock.sent_models[0].params is None
    assert client_api_mock.sent_models[0].metrics == {"eval_loss": 0.25}


def test_submit_model_task_sends_params_without_running_train_or_evaluate(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=1)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, task="submit_model")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    evaluate_result = trainer.evaluate()
    train_result = trainer.train()

    assert evaluate_result is None
    assert train_result is None
    assert trainer.evaluate_call_count == 0
    assert trainer.train_call_count == 0
    assert len(client_api_mock.sent_models) == 1
    assert client_api_mock.sent_models[0].params


def test_submit_model_task_prefers_recorded_checkpoint_params(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=2, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, task="submit_model")
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-2"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "fc.weight": torch.full_like(trainer.model.fc.weight, 11.0),
            "fc.bias": torch.full_like(trainer.model.fc.bias, 12.0),
        },
        checkpoint_dir / "pytorch_model.bin",
    )

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(checkpoint_dir)
    trainer.evaluate()

    sent_params = client_api_mock.sent_models[0].params
    assert torch.equal(sent_params["fc.weight"], torch.full_like(trainer.model.fc.weight, 11.0))
    assert torch.equal(sent_params["fc.bias"], torch.full_like(trainer.model.fc.bias, 12.0))


def test_completed_train_task_does_not_receive_again_before_next_is_running(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer.train()
    evaluate_result = trainer.evaluate()

    assert evaluate_result is None
    assert client_api_mock.receive_calls == 1
    assert len(client_api_mock.sent_models) == 1


def test_default_max_steps_budget_is_captured_before_cumulative_overwrite(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=0, total_rounds=4)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = trainer_cls(
        TinyModel(),
        make_training_args(tmp_path, max_steps=2, per_device_train_batch_size=4, gradient_accumulation_steps=2),
    )

    hf_api.patch(trainer, restore_state=True)
    trainer.train()

    task_state = trainer._nvflare_hf_task_state
    assert task_state.per_round_budget_steps == 2
    assert task_state.cumulative_max_steps == 8
    assert trainer.args.max_steps == 8


def test_cumulative_max_steps_extends_after_checkpoint_resume_restores_global_step(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=5, total_rounds=4)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = trainer_cls(
        TinyModel(),
        make_training_args(tmp_path, max_steps=2, per_device_train_batch_size=4, gradient_accumulation_steps=2),
    )

    hf_api.patch(trainer, restore_state=True)
    task_state = trainer._nvflare_hf_task_state
    task_state.task_kind = hf_api.TASK_TRAIN
    task_state.pending = True
    task_state.per_round_budget_steps = 2
    task_state.cumulative_max_steps = 8
    task_state.last_completed_global_step = 8
    task_state.received_params = _model_params(initial_model, 5.0)
    trainer.state.global_step = 0
    train_kwargs = {}

    task_state._prepare_train_call((), train_kwargs)

    assert task_state.train_start_global_step == 8
    assert task_state.cumulative_max_steps == 10
    assert trainer.args.max_steps == 10


def test_distributed_budget_uses_rank_zero_payload_on_nonzero_rank(monkeypatch, tmp_path):
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[
            {
                "ok": True,
                "operation": "budget capture",
                "per_round_budget_steps": 2,
                "budget_source": "local_epochs",
            },
            {"operation": "resume checkpoint", "checkpoint_path": None},
        ],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = trainer_cls(
        TinyModel(),
        make_training_args(tmp_path, max_steps=-1, num_train_epochs=1, gradient_accumulation_steps=1),
        train_dataloader=[],
    )

    hf_api.patch(trainer, restore_state=True, local_epochs=1.0)
    task_state = trainer._nvflare_hf_task_state
    task_state.task_kind = hf_api.TASK_TRAIN
    task_state.pending = True
    task_state.current_round = 0
    task_state.total_rounds = 4

    task_state._prepare_train_call((), {})

    assert task_state.per_round_budget_steps == 2
    assert task_state.budget_source == "local_epochs"
    assert task_state.cumulative_max_steps == 8
    assert trainer.args.max_steps == 8


def test_epoch_budget_rejects_lengthless_dataloader_at_first_train(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=0, total_rounds=2)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    lengthless_dataloader = iter([object(), object()])
    trainer = trainer_cls(TinyModel(), make_training_args(tmp_path, max_steps=-1), lengthless_dataloader)

    hf_api.patch(trainer, restore_state=False, local_epochs=1)

    with pytest.raises(RuntimeError, match="length-less train dataloader|local_steps"):
        trainer.train()


def test_epoch_budget_rejects_empty_dataloader_with_data_preparation_hint(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=0, total_rounds=2)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = trainer_cls(TinyModel(), make_training_args(tmp_path, max_steps=-1), train_dataloader=[])

    hf_api.patch(trainer, restore_state=False, local_epochs=1)

    with pytest.raises(ValueError, match="training dataloader is empty|dataset size|batch size"):
        trainer.train()


def test_budget_boundary_forces_save_when_restore_state_is_enabled(monkeypatch, tmp_path):
    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=0, total_rounds=2)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    task_state = trainer._nvflare_hf_task_state
    task_state.task_kind = hf_api.TASK_TRAIN
    task_state.pending = True
    task_state.per_round_budget_steps = 1
    task_state.round_stop_step = 1
    trainer.state.global_step = 1

    control = task_state.on_budget_boundary(trainer.state, trainer.control)

    assert control.should_training_stop
    assert control.should_save


def test_checkpoint_injection_strategy_loads_global_params_on_round_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=0, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    trainer.train()

    sent_params = client_api_mock.sent_models[0].params
    assert torch.equal(sent_params["fc.weight"], torch.full_like(sent_params["fc.weight"], 6.0))
    assert torch.equal(sent_params["fc.bias"], torch.full_like(sent_params["fc.bias"], 6.0))


def test_in_memory_resume_reapplies_global_params_after_pre_train_evaluate(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "in_memory")
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, train_with_eval=True)
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    task_state = trainer._nvflare_hf_task_state
    task_state.last_checkpoint_path = str(checkpoint_dir)
    original_train = trainer._train_impl

    def resume_then_train(*args, **kwargs):
        assert kwargs["resume_from_checkpoint"] == str(checkpoint_dir)
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.fill_(2.0)
        return original_train(*args, **kwargs)

    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, resume_then_train)

    trainer.evaluate()
    trainer.train()

    sent_params = client_api_mock.sent_models[0].params
    assert torch.equal(sent_params["fc.weight"], torch.full_like(sent_params["fc.weight"], 6.0))
    assert torch.equal(sent_params["fc.bias"], torch.full_like(sent_params["fc.bias"], 6.0))


def test_checkpoint_injection_writes_global_params_before_resume_on_later_round(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    write_call = {}

    def write_params_to_checkpoint(trainer_arg, checkpoint_dir_arg, params, params_scope, strict=True):
        write_call.update(
            {
                "checkpoint_dir": checkpoint_dir_arg,
                "params": params,
                "params_scope": params_scope,
                "strict": strict,
            }
        )
        hf_api.utils.load_params(trainer_arg, params, params_scope=params_scope, strict=strict, server_key_prefix=None)

    monkeypatch.setattr(hf_api.utils, "write_params_to_checkpoint", write_params_to_checkpoint)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(checkpoint_dir)
    trainer.train()

    assert write_call["checkpoint_dir"] == str(checkpoint_dir)
    assert write_call["params_scope"] == hf_api.utils.PARAMS_SCOPE_MODEL
    assert write_call["strict"] is True
    assert trainer.last_train_kwargs["resume_from_checkpoint"] == str(checkpoint_dir)
    sent_params = client_api_mock.sent_models[0].params
    assert torch.equal(sent_params["fc.weight"], torch.full_like(sent_params["fc.weight"], 6.0))


def test_resume_checkpoint_decision_uses_rank_zero_path_on_nonzero_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    rank_zero_checkpoint = str(tmp_path / "rank-zero-output" / "checkpoint-1")
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[
            {
                "ok": True,
                "operation": "budget capture",
                "per_round_budget_steps": 1,
                "budget_source": "local_steps",
            },
            {"operation": "resume checkpoint", "checkpoint_path": rank_zero_checkpoint},
            {"ok": True, "operation": "checkpoint injection", "error": None},
        ],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    task_state = trainer._nvflare_hf_task_state
    task_state.task_kind = hf_api.TASK_TRAIN
    task_state.pending = True
    task_state.current_round = 1
    task_state.total_rounds = 2
    task_state.last_checkpoint_path = str(tmp_path / "rank-one-local-only" / "checkpoint-1")
    train_kwargs = {}

    task_state._prepare_train_call((), train_kwargs)

    assert train_kwargs["resume_from_checkpoint"] == rank_zero_checkpoint
    assert task_state.global_params_loaded is True
    assert dist.barrier_calls == 1


def test_checkpoint_injection_failure_broadcasts_before_barrier_on_rank_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()

    def write_params_to_checkpoint(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(hf_api.utils, "write_params_to_checkpoint", write_params_to_checkpoint)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(checkpoint_dir)

    with pytest.raises(RuntimeError, match="checkpoint injection failed on rank 0.*disk full"):
        trainer.train()

    status_payloads = [
        payload for payload in dist.broadcast_payloads if payload.get("operation") == "checkpoint injection"
    ]
    assert status_payloads[-1]["ok"] is False
    assert dist.barrier_calls == 0


def test_checkpoint_injection_failure_raises_on_nonzero_rank_before_barrier(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    task_payload = {
        "task_kind": "train",
        "call_name": "train",
        "fl_model": None,
        "params": {},
        "current_round": 1,
        "total_rounds": 2,
    }
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[
            task_payload,
            {
                "ok": True,
                "operation": "budget capture",
                "per_round_budget_steps": 1,
                "budget_source": "local_steps",
            },
            {"operation": "resume checkpoint", "checkpoint_path": str(tmp_path / "checkpoint-1")},
            _rank_zero_failure("checkpoint injection", "OSError: disk full"),
        ],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(checkpoint_dir)

    with pytest.raises(RuntimeError, match="checkpoint injection failed on rank 0.*disk full"):
        trainer.train()

    assert dist.barrier_calls == 0


def test_user_resume_checkpoint_uses_in_memory_override_without_mutating_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY", "checkpoint_injection")
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)
    user_checkpoint_dir = tmp_path / "user-checkpoint"
    user_checkpoint_dir.mkdir()

    def write_params_to_checkpoint(*args, **kwargs):
        raise AssertionError("user-provided checkpoint directory must not be mutated")

    original_train = trainer._train_impl

    def resume_then_train(*args, **kwargs):
        assert kwargs["resume_from_checkpoint"] == str(user_checkpoint_dir)
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.fill_(2.0)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(hf_api.utils, "write_params_to_checkpoint", write_params_to_checkpoint)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, resume_then_train)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(tmp_path / "nvflare-checkpoint")
    trainer.train(resume_from_checkpoint=str(user_checkpoint_dir))

    sent_params = client_api_mock.sent_models[0].params
    assert trainer.last_train_kwargs["resume_from_checkpoint"] == str(user_checkpoint_dir)
    assert torch.equal(sent_params["fc.weight"], torch.full_like(sent_params["fc.weight"], 6.0))


def test_positional_user_resume_checkpoint_is_not_duplicated_as_keyword(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)
    user_checkpoint_dir = tmp_path / "user-checkpoint"
    user_checkpoint_dir.mkdir()
    original_train = trainer._train_impl

    def resume_then_train(*args, **kwargs):
        assert args == (str(user_checkpoint_dir),)
        assert "resume_from_checkpoint" not in kwargs
        return original_train(*args, **kwargs)

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, resume_then_train)
    trainer._nvflare_hf_task_state.last_checkpoint_path = str(tmp_path / "nvflare-checkpoint")
    trainer.train(str(user_checkpoint_dir))

    assert trainer.last_train_args == (str(user_checkpoint_dir),)
    assert "resume_from_checkpoint" not in trainer.last_train_kwargs
    assert len(client_api_mock.sent_models) == 1


def test_receive_agent_closed_maps_to_logged_stop_sentinel(monkeypatch, tmp_path, caplog):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model=None)

    def receive_raises(timeout=None, ctx=None):
        client_api_mock.receive_calls += 1
        raise hf_api.AgentClosed("agent closed")

    client_api_mock.receive = receive_raises
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with caplog.at_level("INFO"):
        result = trainer.evaluate()

    assert result is None
    assert client_api_mock.sent_models == []
    assert "NVFlare job has ended" in caplog.text


def test_receive_protocol_error_propagates_instead_of_stop_sentinel(monkeypatch, tmp_path):
    from nvflare.client.flare_agent import CallStateError

    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model=None)

    def receive_raises(timeout=None, ctx=None):
        client_api_mock.receive_calls += 1
        raise CallStateError("receive called out of order")

    client_api_mock.receive = receive_raises
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(CallStateError, match="out of order"):
        trainer.evaluate()
    assert client_api_mock.sent_models == []


def test_receive_failure_on_rank_zero_broadcasts_before_abort(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model=None)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)

    def receive_raises(timeout=None, ctx=None):
        client_api_mock.receive_calls += 1
        raise RuntimeError("receive boom")

    client_api_mock.receive = receive_raises
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="task dispatch failed on rank 0.*receive boom"):
        trainer.train()

    task_dispatch_payloads = [
        payload for payload in dist.broadcast_payloads if payload.get("operation") == "task dispatch"
    ]
    assert task_dispatch_payloads[-1]["ok"] is False
    assert client_api_mock.sent_models == []
    assert trainer._nvflare_hf_task_state.pending is False


def test_receive_failure_on_rank_zero_reaches_nonzero_rank(monkeypatch, tmp_path):
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=_rank_zero_failure("task dispatch", "RuntimeError: receive boom"),
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="task dispatch failed on rank 0.*receive boom"):
        trainer.train()

    assert trainer._nvflare_hf_task_state.pending is False


def test_is_running_failure_on_rank_zero_broadcasts_and_raises(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model=None)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)

    def is_running_raises(ctx=None):
        raise RuntimeError("is_running boom")

    client_api_mock.is_running = is_running_raises
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="is_running failed on rank 0.*is_running boom"):
        hf_api.hf_is_running()

    assert dist.broadcast_payloads[-1] == _rank_zero_failure("is_running", "RuntimeError: is_running boom")


def test_is_running_failure_on_rank_zero_reaches_nonzero_rank(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=_rank_zero_failure("is_running", "RuntimeError: is_running boom"),
    )
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="is_running failed on rank 0.*is_running boom"):
        hf_api.hf_is_running()


def test_missing_total_rounds_extension_uses_fallback_info_log(monkeypatch, tmp_path, caplog):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=3, total_rounds=None)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=True, local_steps=2)
    task_state = trainer._nvflare_hf_task_state
    task_state.task_kind = hf_api.TASK_TRAIN
    task_state.pending = True
    task_state.per_round_budget_steps = 2
    task_state.cumulative_max_steps = 8
    task_state.train_start_global_step = 8
    task_state.total_rounds = None

    caplog.clear()
    with caplog.at_level(logging.INFO):
        task_state._apply_cumulative_max_steps()

    assert task_state.cumulative_max_steps == 10
    assert trainer.args.max_steps == 10
    assert "FLModel.total_rounds is missing" in caplog.text
    assert "more HuggingFace train rounds than the original total_rounds plan" not in caplog.text
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_aborted_pending_task_blocks_next_is_running_with_actionable_error(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)

    def fail_train(*args, **kwargs):
        raise ValueError("boom")

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    setattr(trainer, hf_api.ORIGINAL_TRAIN_ATTR, fail_train)

    with pytest.raises(ValueError, match="boom"):
        trainer.train()
    with pytest.raises(RuntimeError, match="aborted|restart"):
        hf_api.hf_is_running()


def test_train_send_failure_on_rank_zero_broadcasts_before_abort(monkeypatch, tmp_path):
    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    def send_raises(*args, **kwargs):
        raise RuntimeError("pipe closed")

    client_api_mock.send = send_raises
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="train result send failed on rank 0.*pipe closed"):
        trainer.train()

    status_payloads = [
        payload for payload in dist.broadcast_payloads if payload.get("operation") == "train result send"
    ]
    assert status_payloads[-1]["ok"] is False
    assert trainer._nvflare_hf_task_state.aborted is True


def test_train_result_materialization_failure_on_rank_zero_broadcasts_before_abort(monkeypatch, tmp_path):
    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    def extract_raises(*args, **kwargs):
        raise RuntimeError("extract boom")

    monkeypatch.setattr(hf_api.utils, "extract_params", extract_raises)
    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="train result send failed on rank 0.*extract boom"):
        trainer.train()

    status_payloads = [
        payload for payload in dist.broadcast_payloads if payload.get("operation") == "train result send"
    ]
    assert status_payloads[-1]["ok"] is False
    assert client_api_mock.sent_models == []
    assert trainer._nvflare_hf_task_state.aborted is True


def test_train_send_failure_on_rank_zero_reaches_nonzero_rank(monkeypatch, tmp_path):
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[
            _task_payload("train", "train"),
            {
                "ok": True,
                "operation": "budget capture",
                "per_round_budget_steps": 1,
                "budget_source": "local_steps",
            },
            _rank_zero_failure("train result send", "RuntimeError: pipe closed"),
        ],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="train result send failed on rank 0.*pipe closed"):
        trainer.train()

    assert trainer._nvflare_hf_task_state.aborted is True


def test_train_result_materialization_failure_on_rank_zero_reaches_nonzero_rank(monkeypatch, tmp_path):
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[
            _task_payload("train", "train"),
            {
                "ok": True,
                "operation": "budget capture",
                "per_round_budget_steps": 1,
                "budget_source": "local_steps",
            },
            _rank_zero_failure("train result send", "RuntimeError: extract boom"),
        ],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="train result send failed on rank 0.*extract boom"):
        trainer.train()

    assert trainer._nvflare_hf_task_state.aborted is True


@pytest.mark.parametrize(
    "operation,callable_name",
    [
        ("eval metrics send", "_send_metrics"),
        ("submit model send", "_submit_model"),
    ],
)
def test_rank_zero_send_failures_reach_nonzero_rank_for_eval_and_submit(
    monkeypatch, tmp_path, operation, callable_name
):
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=[_rank_zero_failure(operation, "RuntimeError: pipe closed")],
    )
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    task_state = trainer._nvflare_hf_task_state

    with pytest.raises(RuntimeError, match=f"{operation} failed on rank 0.*pipe closed"):
        if callable_name == "_send_metrics":
            task_state._send_metrics({"eval_loss": 0.25})
        else:
            task_state._submit_model()


def test_submit_model_materialization_failure_on_rank_zero_broadcasts_before_abort(monkeypatch, tmp_path):
    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=1, total_rounds=2)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model, task="submit_model")
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    trainer = _make_trainer(trainer_cls, tmp_path)

    def extract_raises(*args, **kwargs):
        raise RuntimeError("submit extract boom")

    monkeypatch.setattr(hf_api.utils, "extract_params", extract_raises)
    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="submit model send failed on rank 0.*submit extract boom"):
        trainer.train()

    status_payloads = [
        payload for payload in dist.broadcast_payloads if payload.get("operation") == "submit model send"
    ]
    assert status_payloads[-1]["ok"] is False
    assert client_api_mock.sent_models == []
    assert trainer._nvflare_hf_task_state.aborted is True


def test_restore_state_checkpoint_path_is_in_memory_only(monkeypatch, tmp_path):
    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=0, total_rounds=2)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    trainer = _make_trainer(trainer_cls, tmp_path)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()

    hf_api.patch(trainer, restore_state=True, local_steps=1)
    trainer.train()
    task_state = trainer._nvflare_hf_task_state

    assert task_state.last_checkpoint_path == str(checkpoint_dir)
    assert not os.path.exists(os.path.join(tmp_path, "_fl_exchange", "fl_state.json"))

    hf_api._reset_global_state_for_test()
    restored_trainer = _make_trainer(trainer_cls, tmp_path)
    hf_api.patch(restored_trainer, restore_state=True, local_steps=1)

    assert restored_trainer._nvflare_hf_task_state.last_checkpoint_path is None


def test_divergent_trainer_call_across_ranks_fails(monkeypatch, tmp_path):
    payload_from_rank_zero = {
        "task_kind": "train",
        "call_name": "train",
        "fl_model": None,
        "params": {},
        "current_round": 1,
        "total_rounds": 3,
    }
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: _RecordingDist(rank=1, incoming_payload=payload_from_rank_zero))
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="Divergent HuggingFace Trainer call"):
        trainer.evaluate()


def test_params_file_exchange_can_be_forced_for_distributed_dispatch(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    monkeypatch.setenv("NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY", "file")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer.train()

    dispatch_payloads = [payload for payload in dist.broadcast_payloads if payload.get("task_kind") == "train"]
    assert len(dispatch_payloads) == 1
    descriptor = dispatch_payloads[0]["params_exchange"]
    assert dispatch_payloads[0]["params"] is None
    assert not os.path.exists(descriptor["path"])
    assert dist.barrier_calls == 2
    assert len(client_api_mock.sent_models) == 1


def test_params_file_exchange_read_failure_on_nonzero_rank_reaches_rank_zero(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(
        rank=0,
        world_size=2,
        all_gather_payload=[
            {"ok": True, "operation": "params file exchange read", "rank": 0, "error": None},
            {"ok": False, "operation": "params file exchange read", "rank": 1, "error": "RuntimeError: stale NFS"},
        ],
    )
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    monkeypatch.setenv("NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY", "file")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="params file exchange read failed.*rank 1.*stale NFS"):
        trainer.train()

    dispatch_payloads = [payload for payload in dist.broadcast_payloads if payload.get("task_kind") == "train"]
    descriptor = dispatch_payloads[0]["params_exchange"]
    assert not os.path.exists(descriptor["path"])
    assert dist.barrier_calls == 2
    assert client_api_mock.sent_models == []
    assert trainer._nvflare_hf_task_state.aborted is True


def test_params_file_exchange_read_failure_on_nonzero_rank_aborts_locally(monkeypatch, tmp_path):
    descriptor = {"path": str(tmp_path / "missing.safetensors"), "format": "safetensors"}
    payload_from_rank_zero = {
        "task_kind": "train",
        "call_name": "train",
        "fl_model": None,
        "params": None,
        "params_exchange": descriptor,
        "current_round": 1,
        "total_rounds": 3,
    }
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model=None)
    dist = _RecordingDist(
        rank=1,
        world_size=2,
        incoming_payload=payload_from_rank_zero,
        all_gather_payload=[
            {"ok": True, "operation": "params file exchange read", "rank": 0, "error": None},
            {"ok": False, "operation": "params file exchange read", "rank": 1, "error": "RuntimeError: stale NFS"},
        ],
    )
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)

    def read_raises(*args, **kwargs):
        raise RuntimeError("stale NFS")

    monkeypatch.setattr(hf_api.utils, "read_params_exchange_file", read_raises)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="params file exchange read failed.*rank 1.*stale NFS"):
        trainer.train()

    assert dist.barrier_calls == 2
    assert trainer._nvflare_hf_task_state.aborted is True


def test_params_object_exchange_can_be_forced_for_distributed_dispatch(monkeypatch, tmp_path):
    initial_model = TinyModel()
    incoming_model = FLModel(params=_model_params(initial_model, 5.0), current_round=1, total_rounds=3)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch, incoming_model)
    dist = _RecordingDist(rank=0, world_size=2)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: dist)
    monkeypatch.setenv("NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY", "object")
    monkeypatch.setenv("NVFLARE_HF_PARAMS_FILE_EXCHANGE_MIN_BYTES", "0")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)
    trainer.train()

    dispatch_payloads = [payload for payload in dist.broadcast_payloads if payload.get("task_kind") == "train"]
    assert len(dispatch_payloads) == 1
    assert dispatch_payloads[0]["params"]
    assert "params_exchange" not in dispatch_payloads[0]
    assert dist.barrier_calls == 0
