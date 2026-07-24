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

import functools
import logging
import math
import os
import weakref
from importlib import metadata
from typing import Mapping, Optional

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
from nvflare.client import api as flare_api
from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.client.flare_agent import AgentClosed
from nvflare.fuel.utils import fobs

from . import utils
from .callbacks import FLCallback, FLMetricsCallback

FL_META_KEY = "__fl_meta__"
HF_STATE_ATTR = "_nvflare_hf_task_state"
HF_PATCHED_ATTR = "_nvflare_hf_patched"
ORIGINAL_TRAIN_ATTR = "_nvflare_hf_original_train"
ORIGINAL_EVALUATE_ATTR = "_nvflare_hf_original_evaluate"

TASK_TRAIN = "train"
TASK_EVALUATE = "evaluate"
TASK_SUBMIT_MODEL = "submit_model"
TASK_STOP = "stop"

CALL_TRAIN = "train"
CALL_EVALUATE = "evaluate"

STRATEGY_AUTO = "auto"
STRATEGY_IN_MEMORY = "in_memory"
STRATEGY_CHECKPOINT_INJECTION = "checkpoint_injection"
STRATEGY_ENV_VAR = "NVFLARE_HF_WEIGHT_OVERRIDE_STRATEGY"
VERIFIED_TRANSFORMERS_VERSION_MIN = "4.40.0"
# Maintenance rule: advance this upper bound only while the real-transformers CI
# latest job passes; pin it back below the first failing release if HF internals drift.
VERIFIED_TRANSFORMERS_VERSION_MAX_EXCLUSIVE = "6.0.0"
PARAMS_FILE_EXCHANGE_MIN_BYTES_ENV_VAR = "NVFLARE_HF_PARAMS_FILE_EXCHANGE_MIN_BYTES"
DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES = 64 * 1024 * 1024
PARAMS_EXCHANGE_STRATEGY_ENV_VAR = "NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY"
PARAMS_EXCHANGE_STRATEGY_AUTO = "auto"
PARAMS_EXCHANGE_STRATEGY_OBJECT = "object"
PARAMS_EXCHANGE_STRATEGY_FILE = "file"

_ACTIVE_STATE = None


def patch(
    trainer,
    restore_state=True,
    load_state_dict_strict=True,
    params_scope="auto",
    server_key_prefix=None,
    local_epochs=None,
    local_steps=None,
    stream_metrics=False,
):
    trainer_cls = _load_trainer_class()
    if not isinstance(trainer, trainer_cls):
        raise TypeError(f"trainer must be an instance of transformers.Trainer, got {type(trainer)}")

    existing_state = getattr(trainer, HF_STATE_ATTR, None)
    if existing_state is not None and getattr(trainer, HF_PATCHED_ATTR, False):
        _validate_repatch_settings(
            existing_state=existing_state,
            trainer=trainer,
            restore_state=restore_state,
            load_state_dict_strict=load_state_dict_strict,
            params_scope=params_scope,
            server_key_prefix=server_key_prefix,
            local_epochs=local_epochs,
            local_steps=local_steps,
            stream_metrics=stream_metrics,
        )
        return trainer

    active_state = _get_active_state()
    if active_state is not None and active_state.trainer is not trainer:
        raise RuntimeError("only one patched HuggingFace Trainer is supported per process")

    if local_epochs is not None and local_steps is not None:
        raise ValueError("Only one of local_epochs or local_steps can be specified")

    args = getattr(trainer, "args", None)
    if args is None:
        raise ValueError("trainer.args is required")
    if getattr(args, "deepspeed", None):
        raise ValueError("DeepSpeed is not supported by the HuggingFace Client API in design Phase 1")
    if getattr(args, "fsdp", None):
        raise ValueError("FSDP is not supported by the HuggingFace Client API in design Phase 1")
    if restore_state and bool(getattr(args, "save_only_model", False)):
        raise ValueError("save_only_model=True is incompatible with restore_state=True")
    if bool(getattr(args, "load_best_model_at_end", False)):
        raise ValueError("load_best_model_at_end=True is incompatible with FL train tasks")

    resolved_rank = _resolve_rank(trainer)
    if resolved_rank > 0 and _torch_dist() is None:
        raise RuntimeError(
            "HuggingFace Client API resolved rank > 0, but torch.distributed is not initialized. "
            "Launch distributed HF jobs with torchrun so non-zero ranks can participate in NVFlare broadcasts."
        )
    _init_client_api_for_rank(resolved_rank)
    from nvflare.app_opt.pt.decomposers import TensorDecomposer

    fobs.register(TensorDecomposer)

    resolved_scope = utils.resolve_params_scope(trainer, params_scope)
    _default_save_total_limit_if_needed(args, restore_state)

    state = _HFTaskState(
        trainer=trainer,
        rank=resolved_rank,
        restore_state=restore_state,
        load_state_dict_strict=load_state_dict_strict,
        params_scope=resolved_scope,
        server_key_prefix=server_key_prefix,
        local_epochs=local_epochs,
        local_steps=local_steps,
        stream_metrics=stream_metrics,
    )
    state.load_persisted_state()
    _register_callbacks(trainer, state, stream_metrics=stream_metrics)
    _wrap_trainer(trainer, state)
    _set_active_state(state)
    return trainer


def hf_is_running() -> bool:
    state = _get_active_state()
    if state is None:
        return flare_api.is_running()
    return state.is_running()


def _load_trainer_class():
    try:
        from transformers import Trainer
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for nvflare.client.hf.patch(). Install transformers to use this API."
        ) from e
    return Trainer


def _validate_repatch_settings(
    existing_state,
    trainer,
    restore_state,
    load_state_dict_strict,
    params_scope,
    server_key_prefix,
    local_epochs,
    local_steps,
    stream_metrics,
):
    resolved_scope = utils.resolve_params_scope(trainer, params_scope)
    new_settings = {
        "restore_state": bool(restore_state),
        "load_state_dict_strict": bool(load_state_dict_strict),
        "params_scope": resolved_scope,
        "server_key_prefix": server_key_prefix,
        "local_epochs": local_epochs,
        "local_steps": local_steps,
        "stream_metrics": bool(stream_metrics),
    }
    existing_settings = existing_state.patch_settings()
    if new_settings != existing_settings:
        raise RuntimeError(
            "HuggingFace Trainer is already patched with different settings. "
            f"existing={existing_settings}, requested={new_settings}"
        )


def _resolve_rank(trainer) -> int:
    dist = _torch_dist()
    if dist is not None:
        return int(dist.get_rank())

    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    args = getattr(trainer, "args", None)
    process_index = getattr(args, "process_index", None)
    if process_index is not None:
        return int(process_index)
    return 0


def _torch_dist():
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if dist.is_available() and dist.is_initialized():
        return dist
    return None


def _world_size() -> int:
    dist = _torch_dist()
    if dist is None:
        return 1
    return int(dist.get_world_size())


def _transformers_version_is_verified() -> bool:
    version = _transformers_version()
    if not version:
        return False
    try:
        from packaging.version import InvalidVersion, Version

        parsed = Version(version)
        return (
            Version(VERIFIED_TRANSFORMERS_VERSION_MIN) <= parsed < Version(VERIFIED_TRANSFORMERS_VERSION_MAX_EXCLUSIVE)
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "Python package 'packaging' is not installed; cannot verify transformers version for the HuggingFace "
            "in-memory restore strategy. Using checkpoint injection fallback."
        )
        return False
    except InvalidVersion:
        return False


def _transformers_version() -> str:
    try:
        import transformers

        version = str(getattr(transformers, "__version__", "") or "")
    except Exception:
        version = ""
    if version:
        return version
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return ""


def _broadcast_object(obj, src=0):
    dist = _torch_dist()
    if dist is None:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _allow_torch_checkpoint_resume_globals():
    """Allow HF Trainer RNG checkpoints to load with PyTorch's safe-loading defaults."""
    try:
        import importlib

        import numpy as np
        import torch.serialization as torch_serialization
    except Exception:
        return

    add_safe_globals = getattr(torch_serialization, "add_safe_globals", None)
    if not callable(add_safe_globals):
        return

    safe_globals = []
    seen_ids = set()

    def add(obj):
        if obj is not None and id(obj) not in seen_ids:
            safe_globals.append(obj)
            seen_ids.add(id(obj))

    for multiarray_module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            multiarray = importlib.import_module(multiarray_module_name)
            add(getattr(multiarray, "_reconstruct", None))
            break
        except Exception:
            continue
    add(getattr(np, "ndarray", None))
    add(getattr(np, "dtype", None))
    for dtype_name in ("uint32", "int64", "float32", "float64"):
        try:
            add(type(np.dtype(dtype_name)))
        except Exception:
            continue

    if safe_globals:
        add_safe_globals(safe_globals)


def _params_file_exchange_min_bytes() -> int:
    value = os.environ.get(PARAMS_FILE_EXCHANGE_MIN_BYTES_ENV_VAR)
    if value is None:
        return DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Invalid %s=%r; using default %s bytes.",
            PARAMS_FILE_EXCHANGE_MIN_BYTES_ENV_VAR,
            value,
            DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES,
        )
        return DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES
    if threshold < 0:
        logging.getLogger(__name__).warning(
            "Invalid %s=%r; using default %s bytes.",
            PARAMS_FILE_EXCHANGE_MIN_BYTES_ENV_VAR,
            value,
            DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES,
        )
        return DEFAULT_PARAMS_FILE_EXCHANGE_MIN_BYTES
    return threshold


def _params_exchange_strategy() -> str:
    strategy = os.environ.get(PARAMS_EXCHANGE_STRATEGY_ENV_VAR, PARAMS_EXCHANGE_STRATEGY_AUTO).lower()
    valid_strategies = {
        PARAMS_EXCHANGE_STRATEGY_AUTO,
        PARAMS_EXCHANGE_STRATEGY_OBJECT,
        PARAMS_EXCHANGE_STRATEGY_FILE,
    }
    if strategy in valid_strategies:
        return strategy
    logging.getLogger(__name__).warning(
        "Invalid %s=%r; using %s.",
        PARAMS_EXCHANGE_STRATEGY_ENV_VAR,
        strategy,
        PARAMS_EXCHANGE_STRATEGY_AUTO,
    )
    return PARAMS_EXCHANGE_STRATEGY_AUTO


def _task_fl_model_payload(fl_model):
    if fl_model is None:
        return None
    return FLModel(
        metrics=fl_model.metrics,
        start_round=fl_model.start_round,
        current_round=fl_model.current_round,
        total_rounds=fl_model.total_rounds,
        meta=fl_model.meta,
    )


def _init_client_api_for_rank(rank: int):
    ctx = flare_api.default_context
    if ctx is None:
        flare_api.init(rank=str(rank))
        return

    existing_rank = "0" if ctx.rank is None else str(ctx.rank)
    if existing_rank != str(rank):
        raise RuntimeError(
            f"Client API already initialized with rank={existing_rank}, but HuggingFace Trainer resolved rank={rank}"
        )


def _default_save_total_limit_if_needed(args, restore_state: bool):
    if not restore_state:
        return
    if getattr(args, "save_total_limit", None) is None:
        setattr(args, "save_total_limit", 2)
        logging.getLogger(__name__).info("Setting TrainingArguments.save_total_limit=2 for FL resume checkpoints")


def _register_callbacks(trainer, state, stream_metrics: bool):
    callbacks = _get_callbacks(trainer)
    if not any(isinstance(cb, FLCallback) for cb in callbacks):
        _add_callback(trainer, FLCallback(state))
    else:
        for cb in callbacks:
            if isinstance(cb, FLCallback):
                cb.task_state = state

    callbacks = _get_callbacks(trainer)
    if stream_metrics and not any(isinstance(cb, FLMetricsCallback) for cb in callbacks):
        _add_callback(trainer, FLMetricsCallback(state))


def _get_callbacks(trainer):
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(callback_handler, "callbacks", None)
    if callbacks is not None:
        return callbacks
    callbacks = getattr(trainer, "callbacks", None)
    if callbacks is None:
        callbacks = []
        setattr(trainer, "callbacks", callbacks)
    return callbacks


def _add_callback(trainer, callback):
    add_callback = getattr(trainer, "add_callback", None)
    if callable(add_callback):
        add_callback(callback)
        return

    callbacks = _get_callbacks(trainer)
    callbacks.append(callback)


def _wrap_trainer(trainer, state):
    if not hasattr(trainer, ORIGINAL_TRAIN_ATTR):
        setattr(trainer, ORIGINAL_TRAIN_ATTR, trainer.train)
    if not hasattr(trainer, ORIGINAL_EVALUATE_ATTR):
        setattr(trainer, ORIGINAL_EVALUATE_ATTR, trainer.evaluate)

    @functools.wraps(getattr(trainer, ORIGINAL_TRAIN_ATTR))
    def train_wrapper(*args, **kwargs):
        return state.wrapped_train(*args, **kwargs)

    @functools.wraps(getattr(trainer, ORIGINAL_EVALUATE_ATTR))
    def evaluate_wrapper(*args, **kwargs):
        return state.wrapped_evaluate(*args, **kwargs)

    trainer.train = train_wrapper
    trainer.evaluate = evaluate_wrapper
    setattr(trainer, HF_STATE_ATTR, state)
    setattr(trainer, HF_PATCHED_ATTR, True)


def _get_active_state():
    global _ACTIVE_STATE
    if _ACTIVE_STATE is None:
        return None
    state = _ACTIVE_STATE()
    if state is None:
        _ACTIVE_STATE = None
    return state


def _set_active_state(state):
    global _ACTIVE_STATE
    _ACTIVE_STATE = weakref.ref(state)


def _reset_global_state_for_test():
    global _ACTIVE_STATE
    _ACTIVE_STATE = None


class _HFTaskState:
    def __init__(
        self,
        trainer,
        rank: int,
        restore_state: bool,
        load_state_dict_strict: bool,
        params_scope: str,
        server_key_prefix: Optional[str],
        local_epochs,
        local_steps,
        stream_metrics: bool,
    ):
        self.trainer = trainer
        self.rank = int(rank)
        self.world_size = _world_size()
        self.restore_state = bool(restore_state)
        self.load_state_dict_strict = bool(load_state_dict_strict)
        self.params_scope = params_scope
        self.server_key_prefix = server_key_prefix
        self.local_epochs = local_epochs
        self.local_steps = local_steps
        self.stream_metrics = bool(stream_metrics)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.task_kind = None
        self.first_call_name = None
        self.fl_model = None
        self.received_params = {}
        self.current_round = None
        self.total_rounds = None
        self.pending = False
        self.completed = False
        self.completed_task_kind = None
        self.aborted = False
        self._inside_train = False
        self.global_params_loaded = False
        self.capture_evaluation = False
        self.pre_train_metrics = None
        self.eval_task_metrics = None

        self.per_round_budget_steps = None
        self.budget_source = None
        self.cumulative_max_steps = None
        self.round_stop_step = None
        self.train_start_global_step = 0
        self.train_start_num_input_tokens_seen = None
        self.last_completed_global_step = 0
        self.last_checkpoint_path = None
        self.last_completed_round = None
        self.metric_step_offset = 0
        self.result_upload_pending = False
        self.weight_override_strategy = self._select_weight_override_strategy()
        self.train_with_evaluation = self._train_with_evaluation_enabled()
        self._warned_missing_round = False

    def patch_settings(self):
        return {
            "restore_state": self.restore_state,
            "load_state_dict_strict": self.load_state_dict_strict,
            "params_scope": self.params_scope,
            "server_key_prefix": self.server_key_prefix,
            "local_epochs": self.local_epochs,
            "local_steps": self.local_steps,
            "stream_metrics": self.stream_metrics,
        }

    def is_running(self) -> bool:
        if self.pending:
            if self.aborted:
                raise RuntimeError(
                    f"Previous HuggingFace FL task '{self.task_kind}' aborted; restart the training script "
                    "or let the executor report the task failure."
                )
            raise RuntimeError(
                f"Previous HuggingFace FL task '{self.task_kind}' is still pending; "
                f"call trainer.{self._expected_call_for_task()}() before the next flare.is_running()."
            )

        if self.completed:
            self._reset_completed_task()

        running = None
        if self.rank == 0:
            running = flare_api.is_running()
        return bool(_broadcast_object(running, src=0))

    def wrapped_evaluate(self, *args, **kwargs):
        if self._inside_train:
            return self.original_evaluate(*args, **kwargs)
        if self._completed_noop(CALL_EVALUATE):
            return None

        try:
            task_kind = self._ensure_task(CALL_EVALUATE)
            if task_kind == TASK_STOP:
                return None

            if task_kind == TASK_TRAIN:
                self._load_global_params_once()
                self.capture_evaluation = True
                try:
                    result = self.original_evaluate(*args, **kwargs)
                finally:
                    self.capture_evaluation = False
                if self.pre_train_metrics is None:
                    self.pre_train_metrics = _extract_metrics(result)
                return result

            if task_kind == TASK_EVALUATE:
                self._load_global_params_once()
                self.capture_evaluation = True
                try:
                    result = self.original_evaluate(*args, **kwargs)
                finally:
                    self.capture_evaluation = False
                metrics = self.eval_task_metrics or _extract_metrics(result)
                self._send_metrics(metrics)
                self._complete_task()
                return result

            if task_kind == TASK_SUBMIT_MODEL:
                self._submit_model()
                self._complete_task()
                return None

            raise RuntimeError(f"Unsupported HF task kind: {task_kind}")
        except Exception:
            self._abort_task()
            raise

    def wrapped_train(self, *args, **kwargs):
        if self._completed_noop(CALL_TRAIN):
            return None

        try:
            task_kind = self._ensure_task(CALL_TRAIN)
            if task_kind == TASK_STOP:
                return None

            if task_kind == TASK_EVALUATE:
                self.logger.info("Skipping trainer.train() for evaluate task; call trainer.evaluate() to complete it")
                return None

            if task_kind == TASK_SUBMIT_MODEL:
                self._submit_model()
                self._complete_task()
                return None

            if task_kind != TASK_TRAIN:
                raise RuntimeError(f"Unsupported HF task kind: {task_kind}")

            train_kwargs = dict(kwargs)
            self._prepare_train_call(train_kwargs)
            self._inside_train = True
            try:
                return self.original_train(*args, **train_kwargs)
            finally:
                self._inside_train = False
        except Exception:
            self._inside_train = False
            self._abort_task()
            raise

    @property
    def original_train(self):
        return getattr(self.trainer, ORIGINAL_TRAIN_ATTR)

    @property
    def original_evaluate(self):
        return getattr(self.trainer, ORIGINAL_EVALUATE_ATTR)

    def on_train_begin(self, hf_train_state):
        if self.task_kind != TASK_TRAIN or not self.pending:
            return
        global_step = int(getattr(hf_train_state, "global_step", self.train_start_global_step) or 0)
        if self.restore_state:
            self.train_start_global_step = max(global_step, int(self.train_start_global_step or 0))
        else:
            self.train_start_global_step = global_step
        self.train_start_num_input_tokens_seen = _optional_int(getattr(hf_train_state, "num_input_tokens_seen", None))
        self.round_stop_step = self.train_start_global_step + int(self.per_round_budget_steps or 0)
        if self.weight_override_strategy != STRATEGY_CHECKPOINT_INJECTION or not self.global_params_loaded:
            self._load_global_params_once()

    def on_budget_boundary(self, hf_train_state, control):
        if self.task_kind != TASK_TRAIN or not self.pending or self.per_round_budget_steps is None:
            return control
        global_step = int(getattr(hf_train_state, "global_step", 0) or 0)
        if self.round_stop_step is not None and global_step >= self.round_stop_step:
            control.should_training_stop = True
            if self.restore_state:
                control.should_save = True
        return control

    def on_evaluate(self, metrics: dict):
        if self._inside_train:
            return
        if not self.capture_evaluation:
            return
        clean_metrics = _extract_metrics(metrics)
        if self.task_kind == TASK_TRAIN and self.pre_train_metrics is None:
            self.pre_train_metrics = clean_metrics
        elif self.task_kind == TASK_EVALUATE:
            self.eval_task_metrics = clean_metrics

    def on_train_end(self, hf_train_state):
        if self.task_kind != TASK_TRAIN or not self.pending:
            return
        if self.train_with_evaluation and self.pre_train_metrics is None:
            raise RuntimeError("train with evaluation missing training metrics, please remember to call evaluate.")

        end_global_step = int(getattr(hf_train_state, "global_step", 0) or 0)
        end_tokens = _optional_int(getattr(hf_train_state, "num_input_tokens_seen", None))
        meta = self._build_meta(end_global_step=end_global_step, end_tokens=end_tokens)

        step_delta = max(0, end_global_step - int(self.train_start_global_step or 0))
        if not self.restore_state:
            self.metric_step_offset += step_delta
        self.last_completed_global_step = end_global_step
        self.last_completed_round = self.current_round
        self.last_checkpoint_path = self._checkpoint_path_from_state(end_global_step)
        self.result_upload_pending = True
        self._persist_state()

        if self.rank == 0:
            params = utils.extract_params(self.trainer, self.params_scope)
            params = utils.prepare_out_params(
                params,
                self._exchange_format(),
                server_expected_format=self._server_expected_format(),
            )
            params = utils.apply_server_key_prefix(params, self.server_key_prefix)
            output_model = FLModel(
                params=params,
                metrics=self.pre_train_metrics,
                current_round=self.current_round,
                total_rounds=self.total_rounds,
                meta=meta,
            )
            flare_api.send(output_model)

        self.result_upload_pending = False
        self._persist_state()
        self._complete_task()

    def metric_step(self, global_step):
        step = int(global_step or 0)
        if self.restore_state:
            return step
        return self.metric_step_offset + step

    def load_persisted_state(self):
        output_dir = self._output_dir()
        if not output_dir:
            return
        state = None
        if self.rank == 0:
            state = utils.read_checkpoint_state(output_dir, job_id=self._job_id()) or {}
            self._warn_stale_checkpoints_without_provenance(output_dir, state)
        state = _broadcast_object(state, src=0) or {}
        if state:
            self.last_checkpoint_path = state.get("checkpoint_path")
            self.cumulative_max_steps = state.get("cumulative_max_steps")
            self.per_round_budget_steps = state.get("per_round_budget_steps")
            self.last_completed_global_step = int(state.get("last_completed_global_step") or 0)
            self.last_completed_round = state.get("last_completed_round")
            self.metric_step_offset = int(state.get("metric_step_offset") or 0)
            self.result_upload_pending = bool(state.get("result_upload_pending", False))
            self.weight_override_strategy = state.get("weight_override_strategy") or self.weight_override_strategy

    def _ensure_task(self, call_name: str) -> str:
        if self.pending:
            return self.task_kind

        payload = None
        if self.rank == 0:
            try:
                fl_model = flare_api.receive()
            except AgentClosed:
                fl_model = None
            if fl_model is None:
                self.logger.info("Skipping trainer.%s() because NVFlare job has ended", call_name)
                payload = {"task_kind": TASK_STOP, "call_name": call_name}
            else:
                payload = {
                    "task_kind": self._read_task_kind(),
                    "call_name": call_name,
                    "fl_model": _task_fl_model_payload(fl_model),
                    "params": utils.strip_server_key_prefix(fl_model.params, self.server_key_prefix),
                    "current_round": fl_model.current_round,
                    "total_rounds": fl_model.total_rounds,
                }

        payload = self._broadcast_task_payload(payload)
        if payload["call_name"] != call_name:
            raise RuntimeError(
                f"Divergent HuggingFace Trainer call across ranks: rank 0 entered trainer.{payload['call_name']}(), "
                f"but this rank entered trainer.{call_name}()."
            )

        task_kind = payload["task_kind"]
        if task_kind == TASK_STOP:
            self.pending = False
            self.completed = True
            self.completed_task_kind = TASK_STOP
            return TASK_STOP

        self.task_kind = task_kind
        self.first_call_name = call_name
        self.fl_model = payload["fl_model"]
        self.received_params = payload["params"] or {}
        self.current_round = payload["current_round"]
        self.total_rounds = payload["total_rounds"]
        self.pending = True
        self.completed = False
        self.completed_task_kind = None
        self.aborted = False
        self.global_params_loaded = False
        self.capture_evaluation = False
        self.pre_train_metrics = None
        self.eval_task_metrics = None
        return task_kind

    def _broadcast_task_payload(self, payload):
        if self.rank == 0 and payload and payload.get("task_kind") != TASK_STOP:
            params = payload.get("params") or {}
            if self._should_stage_params(params):
                try:
                    descriptor = utils.write_params_exchange_file(self._output_dir(), params)
                    payload = dict(payload)
                    payload["params"] = None
                    payload["params_exchange"] = descriptor
                except Exception as e:
                    self.logger.warning(
                        "Could not stage HuggingFace FL params under %s; using torch.distributed object broadcast. "
                        "Error: %s",
                        utils.get_fl_exchange_dir(self._output_dir()),
                        e,
                    )

        payload = _broadcast_object(payload, src=0)
        descriptor = payload.pop("params_exchange", None) if payload else None
        if descriptor:
            _barrier()
            try:
                payload["params"] = utils.read_params_exchange_file(descriptor)
            finally:
                _barrier()
                if self.rank == 0:
                    utils.cleanup_params_exchange_file(descriptor)
        return payload

    def _should_stage_params(self, params: Mapping) -> bool:
        if self.world_size <= 1 or not params:
            return False
        strategy = _params_exchange_strategy()
        if strategy == PARAMS_EXCHANGE_STRATEGY_OBJECT:
            return False
        if strategy == PARAMS_EXCHANGE_STRATEGY_FILE:
            return True
        return utils.params_nbytes(params) >= _params_file_exchange_min_bytes()

    def _warn_stale_checkpoints_without_provenance(self, output_dir: str, state: dict):
        if not self.restore_state or state:
            return
        checkpoint_dirs = utils.list_checkpoint_dirs(output_dir)
        if not checkpoint_dirs:
            return
        self.logger.warning(
            "Found HuggingFace checkpoint directories under %s but no matching NVFlare checkpoint provenance for "
            "job %r. These checkpoints will not be used for restore_state=True.",
            output_dir,
            self._job_id(),
        )

    def _read_task_kind(self) -> str:
        if flare_api.is_train():
            return TASK_TRAIN
        if flare_api.is_evaluate():
            return TASK_EVALUATE
        if flare_api.is_submit_model():
            return TASK_SUBMIT_MODEL
        raise RuntimeError("Received an unsupported Client API task for the HuggingFace Trainer integration")

    def _prepare_train_call(self, train_kwargs: dict):
        self._capture_budget_if_needed()
        if not self.restore_state:
            self._reset_stateless_trainer_task_state()
        current_global_step = int(getattr(getattr(self.trainer, "state", None), "global_step", 0) or 0)
        if self.restore_state:
            self.train_start_global_step = max(current_global_step, int(self.last_completed_global_step or 0))
        else:
            self.train_start_global_step = current_global_step
        self._apply_cumulative_max_steps()
        self.round_stop_step = self.train_start_global_step + int(self.per_round_budget_steps)

        checkpoint_path = self._resume_checkpoint_path()
        user_resume_checkpoint = (
            train_kwargs.get("resume_from_checkpoint") if "resume_from_checkpoint" in train_kwargs else None
        )
        user_resume_checkpoint_supplied = user_resume_checkpoint is not None
        if user_resume_checkpoint_supplied:
            self.logger.warning(
                "Using user-provided resume_from_checkpoint=%s instead of NVFlare checkpoint provenance. "
                "NVFlare will not modify that checkpoint; received global params will be applied in memory after resume.",
                user_resume_checkpoint,
            )
            checkpoint_path = user_resume_checkpoint

        if checkpoint_path:
            _allow_torch_checkpoint_resume_globals()
            if user_resume_checkpoint_supplied:
                self.global_params_loaded = False
            elif self.weight_override_strategy == STRATEGY_CHECKPOINT_INJECTION:

                def write_checkpoint_params():
                    utils.write_params_to_checkpoint(
                        self.trainer,
                        checkpoint_path,
                        self.received_params,
                        params_scope=self.params_scope,
                        strict=self.load_state_dict_strict,
                    )

                self._run_rank_zero_operation("checkpoint injection", write_checkpoint_params)
                _barrier()
                self.global_params_loaded = True
            elif self.weight_override_strategy == STRATEGY_IN_MEMORY:
                self.global_params_loaded = False

            if not user_resume_checkpoint_supplied:
                train_kwargs["resume_from_checkpoint"] = checkpoint_path

    def _reset_stateless_trainer_task_state(self):
        for attr_name in ("optimizer", "lr_scheduler"):
            if hasattr(self.trainer, attr_name):
                setattr(self.trainer, attr_name, None)
        if hasattr(self.trainer, "_created_lr_scheduler"):
            setattr(self.trainer, "_created_lr_scheduler", False)

        state = getattr(self.trainer, "state", None)
        if state is not None:
            try:
                self.trainer.state = type(state)()
            except Exception:
                for attr_name in ("global_step", "epoch", "num_input_tokens_seen"):
                    if hasattr(state, attr_name):
                        setattr(state, attr_name, 0)

        control = getattr(self.trainer, "control", None)
        if control is not None:
            try:
                self.trainer.control = type(control)()
            except Exception:
                pass

    def _capture_budget_if_needed(self):
        if self.per_round_budget_steps is not None:
            return

        if self.local_steps is not None:
            self.per_round_budget_steps = int(self.local_steps)
            self.budget_source = "local_steps"
        elif self.local_epochs is not None:
            self.per_round_budget_steps = self._epochs_to_steps(float(self.local_epochs))
            self.budget_source = "local_epochs"
        else:
            args = getattr(self.trainer, "args")
            max_steps = int(getattr(args, "max_steps", 0) or 0)
            if max_steps > 0:
                self.per_round_budget_steps = max_steps
                self.budget_source = "args.max_steps"
            else:
                self.per_round_budget_steps = self._epochs_to_steps(float(getattr(args, "num_train_epochs", 1.0)))
                self.budget_source = "args.num_train_epochs"

        if int(self.per_round_budget_steps) <= 0:
            raise ValueError("The HuggingFace local training budget must resolve to at least one optimizer step")

    def _epochs_to_steps(self, local_epochs: float) -> int:
        get_dataloader = getattr(self.trainer, "get_train_dataloader", None)
        if not callable(get_dataloader):
            raise RuntimeError("trainer.get_train_dataloader() is required to convert local_epochs to optimizer steps")
        train_dataloader = get_dataloader()
        try:
            dataloader_len = len(train_dataloader)
        except TypeError as e:
            raise RuntimeError(
                "Cannot convert local_epochs to optimizer steps for a length-less train dataloader; "
                "set local_steps in flare.patch()."
            ) from e

        if dataloader_len == 0:
            raise ValueError(
                "The HuggingFace training dataloader is empty (0 batches); check dataset size, filtering, "
                "batch size, and drop_last configuration."
            )

        grad_accum = max(1, int(getattr(getattr(self.trainer, "args"), "gradient_accumulation_steps", 1) or 1))
        steps_per_epoch = math.ceil(dataloader_len / grad_accum)
        return int(math.ceil(local_epochs * steps_per_epoch))

    def _apply_cumulative_max_steps(self):
        args = getattr(self.trainer, "args")
        if not self.restore_state:
            setattr(args, "max_steps", int(self.per_round_budget_steps))
            return

        if self.cumulative_max_steps is None:
            if self.total_rounds is not None:
                self.cumulative_max_steps = int(self.per_round_budget_steps) * int(self.total_rounds)
            else:
                self.cumulative_max_steps = self.train_start_global_step + int(self.per_round_budget_steps)
                self.logger.warning(
                    "FLModel.total_rounds is missing; extending TrainingArguments.max_steps one round at a time. "
                    "This is safe only for constant learning-rate schedules."
                )
            setattr(args, "max_steps", int(self.cumulative_max_steps))
            return

        self._extend_cumulative_max_steps_if_needed()
        setattr(args, "max_steps", int(self.cumulative_max_steps))

    def _extend_cumulative_max_steps_if_needed(self):
        if not self.restore_state or self.cumulative_max_steps is None:
            return
        if self.train_start_global_step >= int(self.cumulative_max_steps):
            self.cumulative_max_steps = int(self.cumulative_max_steps) + int(self.per_round_budget_steps)
            if self.total_rounds is None:
                self.logger.info(
                    "FLModel.total_rounds is missing; extending TrainingArguments.max_steps to %s for the next "
                    "HuggingFace train round.",
                    self.cumulative_max_steps,
                )
            else:
                self.logger.warning(
                    "Server scheduled more HuggingFace train rounds than the original total_rounds plan; "
                    "extending TrainingArguments.max_steps to %s.",
                    self.cumulative_max_steps,
                )
        setattr(getattr(self.trainer, "args"), "max_steps", int(self.cumulative_max_steps))

    def _load_global_params_once(self):
        if self.global_params_loaded or not self.received_params:
            return
        utils.load_params(
            self.trainer,
            self.received_params,
            params_scope=self.params_scope,
            strict=self.load_state_dict_strict,
            server_key_prefix=None,
        )
        self.global_params_loaded = True

    def _send_metrics(self, metrics: dict):
        if self.rank == 0:
            flare_api.send(FLModel(metrics=metrics, current_round=self.current_round, total_rounds=self.total_rounds))

    def _submit_model(self):
        if self.rank != 0:
            return
        params = None
        if self.last_checkpoint_path:
            params = utils.extract_params_from_checkpoint(self.last_checkpoint_path, self.params_scope)
            if params is None:
                self.logger.warning(
                    "Could not read HuggingFace checkpoint params from %s; submitting current in-memory model params.",
                    self.last_checkpoint_path,
                )
        else:
            self.logger.warning(
                "submit_model requested before any HuggingFace FL train round completed; "
                "submitting current in-memory model parameters."
            )
        if params is None:
            params = utils.extract_params(self.trainer, self.params_scope)
        params = utils.prepare_out_params(
            params,
            self._exchange_format(),
            server_expected_format=self._server_expected_format(),
        )
        params = utils.apply_server_key_prefix(params, self.server_key_prefix)
        flare_api.send(
            FLModel(
                params=params,
                current_round=self.current_round,
                total_rounds=self.total_rounds,
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 0},
            )
        )

    def _complete_task(self):
        self.pending = False
        self.completed = True
        self.completed_task_kind = self.task_kind
        self.task_kind = None
        self.first_call_name = None
        self.fl_model = None
        self.received_params = {}
        self.capture_evaluation = False
        self.global_params_loaded = False

    def _abort_task(self):
        if self.pending:
            self.aborted = True

    def _completed_noop(self, call_name: str) -> bool:
        if not self.completed:
            return False
        self.logger.info("Skipping trainer.%s() because the current FL task is already complete", call_name)
        return True

    def _reset_completed_task(self):
        self.completed = False
        self.completed_task_kind = None
        self.pre_train_metrics = None
        self.eval_task_metrics = None

    def _expected_call_for_task(self) -> str:
        if self.task_kind == TASK_EVALUATE:
            return CALL_EVALUATE
        if self.task_kind == TASK_TRAIN:
            return CALL_TRAIN
        return f"{CALL_TRAIN}() or {CALL_EVALUATE}"

    def _resume_checkpoint_path(self):
        if not self.restore_state:
            return None
        if self.current_round == 0:
            return None
        if self.result_upload_pending and self.current_round == self.last_completed_round:
            if self.rank == 0:
                self.logger.warning(
                    "Ignoring HuggingFace checkpoint provenance from a previous pending result upload for retry "
                    "of round %s.",
                    self.current_round,
                )
            return None
        if self.current_round is None and self.last_checkpoint_path and not self._warned_missing_round:
            self.logger.warning(
                "Received a HuggingFace train task without current_round; resuming from recorded NVFlare checkpoint "
                "because restore_state=True and checkpoint provenance exists."
            )
            self._warned_missing_round = True
        if self.last_checkpoint_path and os.path.isdir(self.last_checkpoint_path):
            return self.last_checkpoint_path
        return None

    def _checkpoint_path_from_state(self, global_step: int):
        output_dir = self._output_dir()
        path = utils.find_checkpoint_for_step(output_dir, global_step)
        if path:
            return path
        if self.restore_state and self.rank == 0:
            self.logger.warning(
                "Expected HuggingFace checkpoint-%s was not found under %s; keeping previous NVFlare checkpoint "
                "provenance. The next round may not restore the latest optimizer/scheduler state.",
                global_step,
                output_dir,
            )
        return self.last_checkpoint_path

    def _persist_state(self):
        output_dir = self._output_dir()

        def write_state():
            if not output_dir:
                return
            utils.write_checkpoint_state(
                output_dir,
                {
                    "job_id": self._job_id(),
                    "world_size": self.world_size,
                    "last_completed_round": self.last_completed_round,
                    "checkpoint_path": self.last_checkpoint_path,
                    "cumulative_max_steps": self.cumulative_max_steps,
                    "per_round_budget_steps": self.per_round_budget_steps,
                    "weight_override_strategy": self.weight_override_strategy,
                    "last_completed_global_step": self.last_completed_global_step,
                    "metric_step_offset": self.metric_step_offset,
                    "result_upload_pending": self.result_upload_pending,
                },
            )

        self._run_rank_zero_operation("checkpoint state persistence", write_state)
        _barrier()

    def _run_rank_zero_operation(self, operation_name: str, operation):
        rank_zero_error = None
        status = {"ok": True, "operation": operation_name, "error": None}
        if self.rank == 0:
            try:
                operation()
            except Exception as e:
                rank_zero_error = e
                status = {"ok": False, "operation": operation_name, "error": f"{type(e).__name__}: {e}"}

        status = _broadcast_object(status, src=0) or {}
        if not status.get("ok", False):
            message = (
                f"HuggingFace distributed {status.get('operation', operation_name)} failed on rank 0: "
                f"{status.get('error', 'unknown error')}"
            )
            if rank_zero_error is not None:
                raise RuntimeError(message) from rank_zero_error
            raise RuntimeError(message)

    def _build_meta(self, end_global_step: int, end_tokens: Optional[int]):
        model = utils.unwrap_model(self.trainer)
        fl_meta = getattr(model, FL_META_KEY, {})
        if fl_meta is None:
            fl_meta = {}
        if not isinstance(fl_meta, dict):
            raise RuntimeError(f"The {FL_META_KEY} attribute must be a dictionary")

        meta = dict(fl_meta)
        if MetaKey.NUM_STEPS_CURRENT_ROUND not in meta:
            args = getattr(self.trainer, "args")
            use_token_count = bool(getattr(args, "include_num_input_tokens_seen", False))
            token_delta = None
            if use_token_count and end_tokens is not None and self.train_start_num_input_tokens_seen is not None:
                token_delta = max(0, end_tokens - self.train_start_num_input_tokens_seen)

            if token_delta is not None and token_delta > 0:
                meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = token_delta
            else:
                step_delta = max(0, end_global_step - int(self.train_start_global_step or 0))
                batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
                grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
                meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = step_delta * batch_size * grad_accum * self.world_size
        return meta

    def _train_with_evaluation_enabled(self) -> bool:
        try:
            return bool(flare_api.get_config().get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRAIN_WITH_EVAL, False))
        except Exception:
            return False

    def _exchange_format(self):
        try:
            return (
                flare_api.get_config()
                .get(ConfigKey.TASK_EXCHANGE, {})
                .get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.PYTORCH)
            )
        except Exception:
            return ExchangeFormat.PYTORCH

    def _server_expected_format(self):
        try:
            return (
                flare_api.get_config()
                .get(ConfigKey.TASK_EXCHANGE, {})
                .get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY)
            )
        except Exception:
            return ExchangeFormat.NUMPY

    def _job_id(self):
        try:
            return flare_api.get_job_id()
        except Exception:
            return ""

    def _output_dir(self):
        return str(getattr(getattr(self.trainer, "args", None), "output_dir", ".") or ".")

    def _select_weight_override_strategy(self) -> str:
        strategy = os.environ.get(STRATEGY_ENV_VAR, STRATEGY_AUTO)
        if strategy not in {STRATEGY_AUTO, STRATEGY_IN_MEMORY, STRATEGY_CHECKPOINT_INJECTION}:
            raise ValueError(
                f"{STRATEGY_ENV_VAR} must be one of {STRATEGY_AUTO}, {STRATEGY_IN_MEMORY}, "
                f"or {STRATEGY_CHECKPOINT_INJECTION}"
            )
        if strategy == STRATEGY_AUTO:
            if _transformers_version_is_verified():
                return STRATEGY_IN_MEMORY
            self.logger.warning(
                "Installed transformers version is outside the verified NVFlare HF in-memory override range "
                "[%s, %s); using checkpoint injection fallback.",
                VERIFIED_TRANSFORMERS_VERSION_MIN,
                VERIFIED_TRANSFORMERS_VERSION_MAX_EXCLUSIVE,
            )
            return STRATEGY_CHECKPOINT_INJECTION
        return strategy


def _barrier():
    dist = _torch_dist()
    if dist is not None:
        dist.barrier()


def _optional_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(metrics):
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        return {}

    result = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        result[key] = value
    return result
