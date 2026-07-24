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

import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
transformers = pytest.importorskip("transformers")

from transformers import Trainer, TrainerCallback, TrainingArguments  # noqa: E402

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey  # noqa: E402
from nvflare.client.config import ConfigKey, ExchangeFormat  # noqa: E402


def _import_real_hf_api_modules():
    for loaded_name in list(sys.modules):
        if loaded_name == "nvflare.client.hf" or loaded_name.startswith("nvflare.app_opt.hf"):
            sys.modules.pop(loaded_name, None)

    import nvflare.app_opt.hf.api as hf_api
    import nvflare.client.hf as flare

    return hf_api, flare


class TinyRegressionDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        x = torch.tensor([float(index), float(index + 1)])
        return {"input_ids": x, "labels": torch.tensor(float(index))}


class TinyRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, input_ids=None, labels=None):
        logits = self.linear(input_ids.float()).squeeze(-1)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = torch.nn.functional.mse_loss(logits, labels.float())
        return result


def _training_args(output_dir, **overrides):
    values = {
        "disable_tqdm": True,
        "logging_strategy": "no",
        "max_steps": 1,
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "report_to": [],
        "save_strategy": "steps",
        "save_steps": 1,
        "save_total_limit": 2,
    }
    values.update(overrides)
    return TrainingArguments(**values)


def _first_param_value(model):
    return float(next(model.parameters()).detach().flatten()[0].cpu())


def _has_model_weights(checkpoint_dir):
    return (checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "pytorch_model.bin").exists()


def test_real_trainer_calls_on_train_begin_after_checkpoint_restore(tmp_path):
    hf_api, _ = _import_real_hf_api_modules()
    train_dataset = TinyRegressionDataset()
    first_model = TinyRegressionModel()
    first_trainer = Trainer(
        model=first_model,
        args=_training_args(tmp_path / "run", max_steps=1),
        train_dataset=train_dataset,
    )
    first_trainer.train()
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    assert checkpoint_dir.is_dir()

    resumed_model = TinyRegressionModel()
    with torch.no_grad():
        for param in resumed_model.parameters():
            param.fill_(99.0)

    seen_at_train_begin = []

    class CaptureTrainBegin(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            seen_at_train_begin.append(_first_param_value(resumed_model))
            return control

    resumed_trainer = Trainer(
        model=resumed_model,
        args=_training_args(tmp_path / "run", max_steps=2, save_strategy="no"),
        train_dataset=train_dataset,
        callbacks=[CaptureTrainBegin()],
    )
    hf_api._allow_torch_checkpoint_resume_globals()
    resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir))

    assert seen_at_train_begin
    assert seen_at_train_begin[0] != 99.0


def test_real_trainer_forced_should_save_writes_resumable_checkpoint(tmp_path):
    class ForceRoundEndSave(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = True
            control.should_training_stop = True
            return control

    trainer = Trainer(
        model=TinyRegressionModel(),
        args=_training_args(tmp_path / "forced", max_steps=5, save_strategy="no"),
        train_dataset=TinyRegressionDataset(),
        callbacks=[ForceRoundEndSave()],
    )
    trainer.train()

    checkpoints = sorted((tmp_path / "forced").glob("checkpoint-*"))
    assert len(checkpoints) == 1
    checkpoint_dir = checkpoints[0]
    assert _has_model_weights(checkpoint_dir)
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert (checkpoint_dir / "scheduler.pt").exists()
    assert (checkpoint_dir / "trainer_state.json").exists()


def test_public_hf_patch_real_trainer_casts_bf16_for_numpy_server(monkeypatch, tmp_path):
    hf_api, flare = _import_real_hf_api_modules()

    hf_api._reset_global_state_for_test()
    sent_models = []
    model = TinyRegressionModel()
    incoming_model = FLModel(params={name: param.detach().clone() for name, param in model.state_dict().items()})

    class CastBFloat16BeforeFLSend(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            model.to(torch.bfloat16)
            return control

    def get_config(ctx=None):
        return {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRAIN_WITH_EVAL: False,
            }
        }

    monkeypatch.setattr(hf_api.flare_api, "default_context", None, raising=False)
    monkeypatch.setattr(hf_api.flare_api, "init", lambda rank=None, config_file=None: None)
    monkeypatch.setattr(hf_api.flare_api, "is_running", lambda ctx=None: True)
    monkeypatch.setattr(hf_api.flare_api, "is_train", lambda ctx=None: True)
    monkeypatch.setattr(hf_api.flare_api, "is_evaluate", lambda ctx=None: False)
    monkeypatch.setattr(hf_api.flare_api, "is_submit_model", lambda ctx=None: False)
    monkeypatch.setattr(hf_api.flare_api, "receive", lambda timeout=None, ctx=None: incoming_model)
    monkeypatch.setattr(hf_api.flare_api, "send", lambda model, clear_cache=True, ctx=None: sent_models.append(model))
    monkeypatch.setattr(hf_api.flare_api, "get_config", get_config)
    monkeypatch.setattr(hf_api.flare_api, "get_job_id", lambda ctx=None: "hf-real-wrapper-job")

    trainer = Trainer(
        model=model,
        args=_training_args(tmp_path / "public-wrapper", max_steps=1, save_strategy="no"),
        train_dataset=TinyRegressionDataset(),
        callbacks=[CastBFloat16BeforeFLSend()],
    )

    flare.patch(trainer, restore_state=False, local_steps=1)
    assert flare.is_running()
    trainer.train()

    assert sent_models
    assert {param.dtype for param in sent_models[0].params.values()} == {torch.float32}

    hf_api._reset_global_state_for_test()


def test_public_hf_patch_restore_state_false_reports_positive_steps_after_state_reset(monkeypatch, tmp_path):
    hf_api, flare = _import_real_hf_api_modules()

    hf_api._reset_global_state_for_test()
    sent_models = []
    model = TinyRegressionModel()
    incoming_models = [
        FLModel(params={name: param.detach().clone() for name, param in model.state_dict().items()}, current_round=0),
        FLModel(params={name: param.detach().clone() for name, param in model.state_dict().items()}, current_round=1),
    ]

    def get_config(ctx=None):
        return {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRAIN_WITH_EVAL: False,
            }
        }

    monkeypatch.setattr(hf_api.flare_api, "default_context", None, raising=False)
    monkeypatch.setattr(hf_api.flare_api, "init", lambda rank=None, config_file=None: None)
    monkeypatch.setattr(hf_api.flare_api, "is_running", lambda ctx=None: True)
    monkeypatch.setattr(hf_api.flare_api, "is_train", lambda ctx=None: True)
    monkeypatch.setattr(hf_api.flare_api, "is_evaluate", lambda ctx=None: False)
    monkeypatch.setattr(hf_api.flare_api, "is_submit_model", lambda ctx=None: False)
    monkeypatch.setattr(hf_api.flare_api, "receive", lambda timeout=None, ctx=None: incoming_models.pop(0))
    monkeypatch.setattr(hf_api.flare_api, "send", lambda model, clear_cache=True, ctx=None: sent_models.append(model))
    monkeypatch.setattr(hf_api.flare_api, "get_config", get_config)
    monkeypatch.setattr(hf_api.flare_api, "get_job_id", lambda ctx=None: "hf-real-two-round-job")

    trainer = Trainer(
        model=model,
        args=_training_args(tmp_path / "two-round-no-restore", max_steps=1, save_strategy="no"),
        train_dataset=TinyRegressionDataset(),
    )

    flare.patch(trainer, restore_state=False, local_steps=1)
    assert flare.is_running()
    trainer.train()
    first_optimizer = trainer.optimizer
    first_lr_scheduler = trainer.lr_scheduler
    assert flare.is_running()
    trainer.train()

    assert len(sent_models) == 2
    assert trainer.optimizer is not None
    assert trainer.lr_scheduler is not None
    assert trainer.optimizer is not first_optimizer
    assert trainer.lr_scheduler is not first_lr_scheduler
    assert sent_models[1].meta[MetaKey.NUM_STEPS_CURRENT_ROUND] > 0

    hf_api._reset_global_state_for_test()
