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

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.client.config import ConfigKey, ExchangeFormat


def install_fake_transformers(monkeypatch):
    """Install the small part of transformers needed by the HF adapter contract tests."""

    module = types.ModuleType("transformers")
    # Keep this inside the verified range so default fake-contract tests exercise the in-memory path.
    module.__version__ = "5.14.1"

    class Trainer:
        pass

    class TrainerCallback:
        pass

    class TrainerControl:
        def __init__(self):
            self.should_save = False
            self.should_training_stop = False

    class TrainerState:
        def __init__(self):
            self.global_step = 0
            self.num_input_tokens_seen = 0

    module.Trainer = Trainer
    module.TrainerCallback = TrainerCallback
    module.TrainerControl = TrainerControl
    module.TrainerState = TrainerState
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def install_fake_peft(monkeypatch):
    """Install a minimal PEFT module with adapter get/set hooks."""

    module = types.ModuleType("peft")

    class PeftModel:
        pass

    def get_peft_model_state_dict(model):
        return model.get_adapter_state_dict()

    def set_peft_model_state_dict(model, state_dict):
        return model.load_adapter_state_dict(state_dict)

    module.PeftModel = PeftModel
    module.get_peft_model_state_dict = get_peft_model_state_dict
    module.set_peft_model_state_dict = set_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "peft", module)
    return module


def import_hf_module(monkeypatch, module_name: str):
    """Import a future HF module, skipping cleanly while the package is absent."""

    install_fake_transformers(monkeypatch)
    for loaded_name in list(sys.modules):
        if loaded_name == "nvflare.client.hf" or loaded_name.startswith("nvflare.app_opt.hf"):
            sys.modules.pop(loaded_name, None)

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        missing_name = err.name or ""
        if module_name == missing_name or module_name.startswith(f"{missing_name}."):
            pytest.skip(f"{module_name} is not implemented yet")
        if missing_name in {"nvflare.client.hf", "nvflare.app_opt.hf"}:
            pytest.skip(f"{module_name} is not implemented yet")
        if missing_name.startswith("nvflare.app_opt.hf."):
            pytest.skip(f"{module_name} depends on {missing_name}, which is not implemented yet")
        raise


def import_hf_utils_module(monkeypatch):
    """Import utils even while the package __init__ is waiting on api.py/callbacks.py."""

    install_fake_transformers(monkeypatch)
    for loaded_name in list(sys.modules):
        if loaded_name.startswith("nvflare.app_opt.hf"):
            sys.modules.pop(loaded_name, None)

    try:
        return importlib.import_module("nvflare.app_opt.hf.utils")
    except ModuleNotFoundError as err:
        missing_name = err.name or ""
        if missing_name not in {"nvflare.app_opt.hf.api", "nvflare.app_opt.hf.callbacks"}:
            if missing_name.startswith("nvflare.app_opt.hf.utils"):
                pytest.skip("nvflare.app_opt.hf.utils is not implemented yet")
            raise

    repo_root = Path(__file__).resolve().parents[4]
    utils_path = repo_root / "nvflare" / "app_opt" / "hf" / "utils.py"
    if not utils_path.exists():
        pytest.skip("nvflare.app_opt.hf.utils is not implemented yet")

    spec = importlib.util.spec_from_file_location("nvflare_app_opt_hf_utils_contract_under_test", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ClientAPIMock:
    def __init__(
        self,
        incoming_model=None,
        task: str = "train",
        exchange_format: ExchangeFormat = ExchangeFormat.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        train_with_eval: bool = False,
        running: bool = True,
    ):
        self.incoming_models = [incoming_model] if incoming_model is not None else []
        self.task = task
        self.running = running
        self.init_calls = []
        self.receive_calls = 0
        self.sent_models = []
        self.events = []
        self.config = {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: exchange_format,
                ConfigKey.SERVER_EXPECTED_FORMAT: server_expected_format,
                ConfigKey.TRAIN_WITH_EVAL: train_with_eval,
            }
        }

    def init(self, rank=None, config_file=None):
        self.events.append("init")
        self.init_calls.append({"rank": rank, "config_file": config_file})
        return SimpleNamespace(rank=str(rank) if rank is not None else None, config_file=config_file)

    def receive(self, timeout=None, ctx=None):
        self.events.append("receive")
        self.receive_calls += 1
        if self.incoming_models:
            return self.incoming_models.pop(0)
        return None

    def send(self, model, clear_cache=True, ctx=None):
        self.events.append("send")
        self.sent_models.append(model)

    def get_config(self, ctx=None):
        return self.config

    def get_job_id(self, ctx=None):
        return "hf-contract-job"

    def get_site_name(self, ctx=None):
        return "site-1"

    def get_task_name(self, ctx=None):
        return self.task

    def system_info(self, ctx=None):
        return {"job_id": self.get_job_id(), "site_name": self.get_site_name()}

    def is_running(self, ctx=None):
        self.events.append("is_running")
        return self.running

    def is_train(self, ctx=None):
        return self.task == "train"

    def is_evaluate(self, ctx=None):
        return self.task == "evaluate"

    def is_submit_model(self, ctx=None):
        return self.task == "submit_model"

    def clear(self, ctx=None):
        self.events.append("clear")

    def shutdown(self, ctx=None):
        self.events.append("shutdown")

    def log(self, *args, **kwargs):
        self.events.append("log")
        return True


def patch_client_api_aliases(monkeypatch, client_api_mock: ClientAPIMock, *modules):
    """Patch common Client API import styles used by adapter code."""

    import nvflare.client as client_package
    import nvflare.client.api as client_api

    patched = set()

    def patch_target(target):
        if target is None or id(target) in patched:
            return
        patched.add(id(target))
        for name in (
            "clear",
            "get_config",
            "get_job_id",
            "get_site_name",
            "get_task_name",
            "init",
            "is_evaluate",
            "is_running",
            "is_submit_model",
            "is_train",
            "log",
            "receive",
            "send",
            "shutdown",
            "system_info",
        ):
            if hasattr(target, name):
                monkeypatch.setattr(target, name, getattr(client_api_mock, name), raising=False)

    for target in (client_package, client_api, *modules):
        patch_target(target)
        for attr_name in ("api", "client_api", "_client_api", "flare"):
            patch_target(getattr(target, attr_name, None))


class FakeAccelerator:
    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return model.state_dict()


def make_training_args(output_dir, **overrides):
    values = {
        "bf16": False,
        "deepspeed": None,
        "fsdp": None,
        "gradient_accumulation_steps": 1,
        "include_num_input_tokens_seen": False,
        "load_best_model_at_end": False,
        "local_rank": -1,
        "max_steps": -1,
        "num_train_epochs": 1,
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "process_index": 0,
        "save_only_model": False,
        "save_strategy": "no",
        "save_total_limit": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def call_hf_callback(callback, hook_name, trainer, **kwargs):
    hook = getattr(callback, hook_name, None)
    if hook:
        return hook(trainer.args, trainer.state, trainer.control, model=trainer.model, trainer=trainer, **kwargs)
    return None


def make_fake_trainer_class(transformers_module):
    class FakeTrainer(transformers_module.Trainer):
        def __init__(self, model, args, train_dataloader=None):
            self.model = model
            self.model_wrapped = model
            self.args = args
            self.accelerator = FakeAccelerator()
            self.callbacks = []
            self.callback_handler = SimpleNamespace(callbacks=self.callbacks)
            self.control = transformers_module.TrainerControl()
            self.state = transformers_module.TrainerState()
            self.train_call_count = 0
            self.evaluate_call_count = 0
            self.last_train_args = None
            self.last_train_kwargs = None
            self.evaluate_metrics = {"eval_loss": 0.25}
            self.train_dataloader = train_dataloader if train_dataloader is not None else [object(), object()]
            self.train = self._train_impl
            self.evaluate = self._evaluate_impl

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def get_train_dataloader(self):
            return self.train_dataloader

        def _train_impl(self, *args, **kwargs):
            self.train_call_count += 1
            self.last_train_args = args
            self.last_train_kwargs = dict(kwargs)
            for callback in list(self.callbacks):
                call_hf_callback(callback, "on_train_begin", self)
            if hasattr(self.model, "bump_after_train"):
                self.model.bump_after_train()
            self.state.global_step += 1
            for callback in list(self.callbacks):
                call_hf_callback(callback, "on_train_end", self)
            return SimpleNamespace(global_step=self.state.global_step)

        def _evaluate_impl(self, *args, **kwargs):
            self.evaluate_call_count += 1
            metrics = dict(self.evaluate_metrics)
            for callback in list(self.callbacks):
                call_hf_callback(callback, "on_evaluate", self, metrics=metrics)
            return metrics

    return FakeTrainer
