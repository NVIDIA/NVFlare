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

import importlib
import os
import platform
import queue
import socket
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

dist_available = torch.distributed.is_available()
gloo_available = dist_available and torch.distributed.is_gloo_available()
running_under_xdist = bool(os.environ.get("PYTEST_XDIST_WORKER"))
GLOO_INIT_ERROR_MARKERS = (
    "Cannot resolve 127.0.0.1 to a (local) address",
    "Unable to resolve hostname",
    "makeDeviceForInterface(): unsupported gloo device",
    "uv_bind: operation not permitted",
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def bump_after_train(self):
        with torch.no_grad():
            self.fc.weight.add_(1.0)
            self.fc.bias.add_(1.0)


def _model_params(model, value):
    return {name: torch.full_like(param, value) for name, param in model.state_dict().items()}


def _install_fake_transformers():
    module = types.ModuleType("transformers")
    module.__version__ = "4.45.0"

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
    sys.modules["transformers"] = module
    return module


def _call_callback(callback, hook_name, trainer, **kwargs):
    hook = getattr(callback, hook_name, None)
    if hook:
        return hook(trainer.args, trainer.state, trainer.control, model=trainer.model, trainer=trainer, **kwargs)
    return None


def _make_training_args(output_dir):
    return SimpleNamespace(
        bf16=False,
        deepspeed=None,
        fsdp=None,
        gradient_accumulation_steps=1,
        include_num_input_tokens_seen=False,
        load_best_model_at_end=False,
        local_rank=-1,
        max_steps=-1,
        num_train_epochs=1,
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        process_index=0,
        save_only_model=False,
        save_strategy="no",
        save_total_limit=None,
    )


def _make_fake_trainer_class(transformers_module):
    class FakeTrainer(transformers_module.Trainer):
        def __init__(self, model, args):
            self.model = model
            self.model_wrapped = model
            self.args = args
            self.accelerator = SimpleNamespace(
                unwrap_model=lambda model: model, get_state_dict=lambda model: model.state_dict()
            )
            self.callbacks = []
            self.callback_handler = SimpleNamespace(callbacks=self.callbacks)
            self.control = transformers_module.TrainerControl()
            self.state = transformers_module.TrainerState()
            self.train = self._train_impl
            self.evaluate = self._evaluate_impl

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def get_train_dataloader(self):
            return [object(), object()]

        def _train_impl(self, *args, **kwargs):
            for callback in list(self.callbacks):
                _call_callback(callback, "on_train_begin", self)
            self.model.bump_after_train()
            self.state.global_step += 1
            for callback in list(self.callbacks):
                _call_callback(callback, "on_train_end", self)
            return SimpleNamespace(global_step=self.state.global_step)

        def _evaluate_impl(self, *args, **kwargs):
            metrics = {"eval_loss": 0.25}
            for callback in list(self.callbacks):
                _call_callback(callback, "on_evaluate", self, metrics=metrics)
            return metrics

    return FakeTrainer


def _configure_client_api(hf_api, rank, task, result_queue):
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.client.config import ConfigKey, ExchangeFormat

    incoming_model = FLModel(params=_model_params(TinyModel(), 5.0), current_round=1, total_rounds=3)
    sent_models = []

    def init(rank=None, config_file=None):
        return SimpleNamespace(rank=str(rank) if rank is not None else None, config_file=config_file)

    def receive(timeout=None, ctx=None):
        if rank != 0:
            raise AssertionError("nonzero ranks must not call Client API receive")
        return incoming_model

    def send(model, clear_cache=True, ctx=None):
        if rank != 0:
            raise AssertionError("nonzero ranks must not call Client API send")
        sent_models.append(model)

    def get_config(ctx=None):
        return {ConfigKey.TASK_EXCHANGE: {ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH}}

    hf_api.flare_api.default_context = None
    hf_api.flare_api.init = init
    hf_api.flare_api.receive = receive
    hf_api.flare_api.send = send
    hf_api.flare_api.get_config = get_config
    hf_api.flare_api.get_job_id = lambda ctx=None: "hf-gloo-contract-job"
    hf_api.flare_api.is_running = lambda ctx=None: True
    hf_api.flare_api.is_train = lambda ctx=None: task == "train"
    hf_api.flare_api.is_evaluate = lambda ctx=None: task == "evaluate"
    hf_api.flare_api.is_submit_model = lambda ctx=None: task == "submit_model"
    return sent_models


def _set_gloo_loopback_if_needed():
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return
    if platform.system() != "Linux":
        return

    try:
        interface_names = {name for _, name in socket.if_nameindex()}
    except (AttributeError, OSError):
        return

    if "lo" in interface_names:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"


def _is_gloo_init_error(result):
    error = result.get("error", "")
    return any(marker in error for marker in GLOO_INIT_ERROR_MARKERS)


def _worker(rank, init_path, output_dir, strategy, scenario, result_queue):
    import torch.distributed as dist

    try:
        _set_gloo_loopback_if_needed()
        dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
        _install_fake_transformers()
        for loaded_name in list(sys.modules):
            if loaded_name == "nvflare.client.hf" or loaded_name.startswith("nvflare.app_opt.hf"):
                sys.modules.pop(loaded_name, None)
        hf_api = importlib.import_module("nvflare.app_opt.hf.api")
        sent_models = _configure_client_api(hf_api, rank, "train", result_queue)
        os.environ["NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY"] = strategy
        trainer_cls = _make_fake_trainer_class(sys.modules["transformers"])
        trainer = trainer_cls(TinyModel(), _make_training_args(output_dir))
        hf_api.patch(trainer, restore_state=False, local_steps=1)

        if scenario == "receive_failure" and rank == 0:

            def receive_raises(timeout=None, ctx=None):
                raise RuntimeError("rank-zero receive boom")

            hf_api.flare_api.receive = receive_raises

        if scenario == "extract_failure" and rank == 0:

            def extract_raises(*args, **kwargs):
                raise RuntimeError("rank-zero extract boom")

            hf_api.utils.extract_params = extract_raises

        if scenario == "file_read_failure" and rank == 1:

            def read_params_raises(*args, **kwargs):
                raise RuntimeError("rank-one read boom")

            hf_api.utils.read_params_exchange_file = read_params_raises

        if scenario == "divergent":
            if rank == 0:
                trainer._nvflare_hf_task_state._ensure_task(hf_api.CALL_TRAIN)
                dist.barrier()
                result_queue.put({"rank": rank, "ok": True, "sent_count": 0})
            else:
                ok = False
                error = None
                try:
                    trainer.evaluate()
                except RuntimeError as e:
                    ok = "Divergent HuggingFace Trainer call" in str(e)
                    error = str(e)
                else:
                    error = "expected divergent-call failure"
                dist.barrier()
                result = {"rank": rank, "ok": ok}
                if error:
                    result["error"] = error
                result_queue.put(result)
        elif scenario == "extract_failure":
            ok = False
            error = None
            try:
                trainer.train()
            except RuntimeError as e:
                error = str(e)
                ok = "train result send failed on rank 0" in error and "rank-zero extract boom" in error
            else:
                error = "expected rank-zero extraction failure"
            result_queue.put({"rank": rank, "ok": ok, "error": error, "sent_count": len(sent_models)})
        elif scenario == "receive_failure":
            ok = False
            error = None
            try:
                trainer.train()
            except RuntimeError as e:
                error = str(e)
                ok = "task dispatch failed on rank 0" in error and "rank-zero receive boom" in error
            else:
                error = "expected rank-zero receive failure"
            result_queue.put({"rank": rank, "ok": ok, "error": error, "sent_count": len(sent_models)})
        elif scenario == "file_read_failure":
            ok = False
            error = None
            try:
                trainer.train()
            except RuntimeError as e:
                error = str(e)
                ok = "params file exchange read failed" in error and "rank 1" in error and "rank-one read boom" in error
            else:
                error = "expected rank-one file-read failure"
            result_queue.put({"rank": rank, "ok": ok, "error": error, "sent_count": len(sent_models)})
        else:
            trainer.train()
            first_weight = float(trainer.model.fc.weight.detach().cpu().reshape(-1)[0].item())
            result_queue.put(
                {
                    "rank": rank,
                    "ok": True,
                    "first_weight": first_weight,
                    "sent_count": len(sent_models),
                }
            )
    except Exception as e:
        result_queue.put({"rank": rank, "ok": False, "error": repr(e)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_2process_scenario(tmp_path, strategy, scenario):
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    init_path = tmp_path / f"{scenario}-{strategy}.init"
    output_dir = tmp_path / f"{scenario}-{strategy}-output"
    procs = [
        ctx.Process(target=_worker, args=(rank, init_path, output_dir, strategy, scenario, result_queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    assert all(proc.exitcode == 0 for proc in procs)

    results = []
    for _ in procs:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            pytest.fail("timed out waiting for distributed HF worker result")
    if results and all(_is_gloo_init_error(result) for result in results):
        pytest.skip(f"torch.distributed gloo could not initialize on this runner: {results}")
    return sorted(results, key=lambda result: result["rank"])


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required for distributed HF contract tests")
@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing gloo tests are unstable under xdist")
def test_two_process_gloo_file_exchange_dispatch_and_rank_zero_send(tmp_path):
    results = _run_2process_scenario(tmp_path, strategy="file", scenario="train")

    assert all(result["ok"] for result in results)
    assert [result["sent_count"] for result in results] == [1, 0]
    assert [result["first_weight"] for result in results] == [6.0, 6.0]


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required for distributed HF contract tests")
@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing gloo tests are unstable under xdist")
def test_two_process_gloo_detects_divergent_trainer_calls(tmp_path):
    results = _run_2process_scenario(tmp_path, strategy="object", scenario="divergent")

    assert all(result["ok"] for result in results), results
    assert [result["sent_count"] for result in results if result["rank"] == 0] == [0]


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required for distributed HF contract tests")
@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing gloo tests are unstable under xdist")
def test_two_process_gloo_propagates_rank_zero_result_materialization_failure(tmp_path):
    results = _run_2process_scenario(tmp_path, strategy="object", scenario="extract_failure")

    assert all(result["ok"] for result in results), results
    assert [result["sent_count"] for result in results] == [0, 0]


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required for distributed HF contract tests")
@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing gloo tests are unstable under xdist")
def test_two_process_gloo_propagates_file_exchange_read_failure(tmp_path):
    results = _run_2process_scenario(tmp_path, strategy="file", scenario="file_read_failure")

    assert all(result["ok"] for result in results), results
    assert [result["sent_count"] for result in results] == [0, 0]


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required for distributed HF contract tests")
@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing gloo tests are unstable under xdist")
def test_two_process_gloo_propagates_rank_zero_receive_failure(tmp_path):
    results = _run_2process_scenario(tmp_path, strategy="object", scenario="receive_failure")

    assert all(result["ok"] for result in results), results
    assert [result["sent_count"] for result in results] == [0, 0]
