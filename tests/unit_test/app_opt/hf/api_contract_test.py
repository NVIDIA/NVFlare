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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ._helpers import (  # noqa: E402
    ClientAPIMock,
    import_hf_module,
    install_fake_peft,
    make_fake_trainer_class,
    make_training_args,
    patch_client_api_aliases,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)


def _fresh_api(monkeypatch):
    client_api_mock = ClientAPIMock()
    patch_client_api_aliases(monkeypatch, client_api_mock)
    hf_api = import_hf_module(monkeypatch, "nvflare.app_opt.hf.api")
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    transformers = pytest.importorskip("transformers")
    trainer_cls = make_fake_trainer_class(transformers)
    return hf_api, trainer_cls, client_api_mock


def _make_trainer(trainer_cls, tmp_path, model=None, **arg_overrides):
    return trainer_cls(model or TinyModel(), make_training_args(tmp_path, **arg_overrides))


def _fl_callbacks(trainer):
    return [callback for callback in trainer.callbacks if callback.__class__.__name__ == "FLCallback"]


class _FakeDist:
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.barrier_calls = 0

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def broadcast_object_list(self, payload, src=0):
        return None

    def barrier(self):
        self.barrier_calls += 1


def test_patch_rejects_both_explicit_local_budget_styles(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises((RuntimeError, ValueError), match=r"local_epochs.*local_steps|local_steps.*local_epochs|both"):
        hf_api.patch(trainer, local_epochs=1, local_steps=1)


@pytest.mark.parametrize(
    "arg_name,arg_value,match",
    [
        ("deepspeed", "ds_config.json", "DeepSpeed|deepspeed|Phase 2"),
        ("fsdp", "full_shard", "FSDP|fsdp|Phase 2"),
    ],
)
def test_patch_rejects_design_phase_one_unsupported_backends(monkeypatch, tmp_path, arg_name, arg_value, match):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path, **{arg_name: arg_value})

    with pytest.raises((RuntimeError, ValueError), match=match):
        hf_api.patch(trainer, restore_state=False)


@pytest.mark.parametrize(
    "arg_name,match",
    [
        ("save_only_model", "save_only_model|optimizer|scheduler|restore_state"),
        ("load_best_model_at_end", "load_best_model_at_end|best model|server-side model selection"),
    ],
)
def test_patch_rejects_incompatible_training_arguments(monkeypatch, tmp_path, arg_name, match):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path, **{arg_name: True})

    with pytest.raises((RuntimeError, ValueError), match=match):
        hf_api.patch(trainer)


@pytest.mark.parametrize("prebuilt_attr", ["optimizer", "lr_scheduler"])
def test_stateless_mode_rejects_prebuilt_optimizer_or_scheduler(monkeypatch, tmp_path, prebuilt_attr):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)
    prebuilt = object()
    setattr(trainer, prebuilt_attr, prebuilt)

    with pytest.raises(ValueError, match="restore_state=False.*prebuilt Trainer optimizer or scheduler"):
        hf_api.patch(trainer, restore_state=False)

    assert getattr(trainer, prebuilt_attr) is prebuilt


def test_restore_state_rejects_explicit_launch_once_false(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch)
    client_api_mock.config[hf_api.ConfigKey.TASK_EXCHANGE][hf_api.ConfigKey.LAUNCH_ONCE] = False
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises(RuntimeError, match="restore_state=True.*single trainer process|launch_once=True"):
        hf_api.patch(trainer, restore_state=True)


def test_stateless_mode_accepts_explicit_launch_once_false(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch)
    client_api_mock.config[hf_api.ConfigKey.TASK_EXCHANGE][hf_api.ConfigKey.LAUNCH_ONCE] = False
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False)

    assert trainer._nvflare_hf_task_state.restore_state is False


def test_patch_uses_global_rank_and_ignores_local_rank(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: _FakeDist(rank=3, world_size=8))
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("RANK", "3")
    trainer = _make_trainer(trainer_cls, tmp_path, process_index=9)

    hf_api.patch(trainer, restore_state=False)

    assert client_api_mock.init_calls
    assert str(client_api_mock.init_calls[0]["rank"]) == "3"


def test_patch_rejects_nonzero_rank_without_initialized_distributed(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: None)
    monkeypatch.setenv("RANK", "3")
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises(RuntimeError, match="rank > 0|torch.distributed|torchrun"):
        hf_api.patch(trainer, restore_state=False)


def test_patch_rejects_rank_zero_multirank_env_without_initialized_distributed(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: None)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises(RuntimeError, match="WORLD_SIZE|torch.distributed|torchrun"):
        hf_api.patch(trainer, restore_state=False)


def test_patch_rejects_adapter_scope_for_non_peft_model(monkeypatch, tmp_path):
    install_fake_peft(monkeypatch)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises((RuntimeError, ValueError), match="adapter|PEFT|PeftModel"):
        hf_api.patch(trainer, params_scope="adapter")


def test_patch_registers_one_fl_callback_and_wraps_trainer_methods_idempotently(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)
    original_train = trainer.train
    original_evaluate = trainer.evaluate

    hf_api.patch(trainer, restore_state=False)
    first_patched_train = trainer.train
    first_patched_evaluate = trainer.evaluate
    hf_api.patch(trainer, restore_state=False)

    assert len(_fl_callbacks(trainer)) == 1
    assert trainer.train is first_patched_train
    assert trainer.evaluate is first_patched_evaluate
    assert trainer.train is not original_train
    assert trainer.evaluate is not original_evaluate


def test_patch_defaults_save_total_limit_for_restore_state_without_overriding_user_value(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    defaulted = _make_trainer(trainer_cls, tmp_path / "defaulted", save_strategy="steps", save_total_limit=None)
    preserved = _make_trainer(trainer_cls, tmp_path / "preserved", save_strategy="epoch", save_total_limit=5)

    hf_api.patch(defaulted, restore_state=True)
    assert defaulted.args.save_total_limit == 2

    hf_api._reset_global_state_for_test()
    hf_api.patch(preserved, restore_state=True)
    assert preserved.args.save_total_limit == 5


def test_repatch_with_different_settings_raises(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False, local_steps=1)

    with pytest.raises(RuntimeError, match="already patched with different settings"):
        hf_api.patch(trainer, restore_state=False, local_steps=2)


def test_patch_rejects_a_second_trainer_in_the_same_process(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    first_trainer = _make_trainer(trainer_cls, tmp_path / "first")
    second_trainer = _make_trainer(trainer_cls, tmp_path / "second")

    hf_api.patch(first_trainer, restore_state=False)

    with pytest.raises((RuntimeError, ValueError), match="one patched|single patched|already patched"):
        hf_api.patch(second_trainer, restore_state=False)


def test_patch_rejects_mismatched_existing_client_api_rank(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch)
    patch_client_api_aliases(monkeypatch, client_api_mock, hf_api)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: _FakeDist(rank=1, world_size=2))
    monkeypatch.setattr(hf_api.flare_api, "default_context", SimpleNamespace(rank="2"), raising=False)
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises(RuntimeError, match="rank 2|rank.*1|initialized"):
        hf_api.patch(trainer, restore_state=False)


def test_patch_resolves_rank_zero_from_trainer_args_when_no_distributed_or_rank_env(monkeypatch, tmp_path):
    hf_api, trainer_cls, client_api_mock = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_torch_dist", lambda: None)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    trainer = _make_trainer(trainer_cls, tmp_path, process_index=0)

    hf_api.patch(trainer, restore_state=False)

    assert str(client_api_mock.init_calls[0]["rank"]) == "0"


def test_auto_strategy_uses_in_memory_only_for_verified_transformers_version(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_transformers_version", lambda: hf_api.VERIFIED_TRANSFORMERS_VERSION_MIN)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False)

    assert trainer._nvflare_hf_task_state.weight_override_strategy == hf_api.STRATEGY_IN_MEMORY


def test_auto_strategy_uses_in_memory_for_verified_transformers_five_line(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_transformers_version", lambda: "5.14.1")
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False)

    assert trainer._nvflare_hf_task_state.weight_override_strategy == hf_api.STRATEGY_IN_MEMORY


def test_auto_strategy_falls_back_for_unverified_transformers_version(monkeypatch, tmp_path):
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)
    monkeypatch.setattr(hf_api, "_transformers_version", lambda: hf_api.VERIFIED_TRANSFORMERS_VERSION_MAX_EXCLUSIVE)
    trainer = _make_trainer(trainer_cls, tmp_path)

    hf_api.patch(trainer, restore_state=False)

    assert trainer._nvflare_hf_task_state.weight_override_strategy == hf_api.STRATEGY_CHECKPOINT_INJECTION


def test_patch_accepts_peft_auto_scope_after_trainer_construction(monkeypatch, tmp_path):
    peft = install_fake_peft(monkeypatch)
    hf_api, trainer_cls, _ = _fresh_api(monkeypatch)

    class TinyPeftModel(TinyModel, peft.PeftModel):
        def get_adapter_state_dict(self):
            return {"adapter.weight": torch.ones(1)}

        def load_adapter_state_dict(self, state_dict):
            self.loaded_adapter_state = state_dict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    trainer = _make_trainer(trainer_cls, tmp_path, model=TinyPeftModel())

    hf_api.patch(trainer, params_scope="auto", restore_state=False)

    assert len(_fl_callbacks(trainer)) == 1
