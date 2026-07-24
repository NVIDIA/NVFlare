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

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from typing import Optional

from nvflare.client.config import ExchangeFormat

PARAMS_SCOPE_AUTO = "auto"
PARAMS_SCOPE_MODEL = "model"
PARAMS_SCOPE_ADAPTER = "adapter"
VALID_PARAMS_SCOPES = {PARAMS_SCOPE_AUTO, PARAMS_SCOPE_MODEL, PARAMS_SCOPE_ADAPTER}

FL_EXCHANGE_DIR = "_fl_exchange"
PARAMS_EXCHANGE_FORMAT_SAFETENSORS = "safetensors"
PARAMS_EXCHANGE_FORMAT_TORCH = "torch"

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required for nvflare.app_opt.hf") from e
    return torch


def _import_peft(reason: str = "PEFT parameter handling"):
    try:
        import peft
    except ImportError as e:
        raise RuntimeError(f"PEFT is required for {reason}. Install the optional 'peft' dependency.") from e
    return peft


def unwrap_model(trainer):
    model = getattr(trainer, "model", None)
    if model is None:
        raise ValueError("trainer must expose a model attribute.")

    accelerator = getattr(trainer, "accelerator", None)
    unwrap = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap):
        try:
            return unwrap(model)
        except TypeError:
            return unwrap(model, keep_fp32_wrapper=False)
    return model


def _is_peft_class(cls) -> bool:
    return cls.__name__ == "PeftModel" and cls.__module__.split(".", 1)[0] == "peft"


def is_peft_model(model) -> bool:
    if model is None:
        return False

    if any(_is_peft_class(cls) for cls in type(model).__mro__):
        return True

    try:
        peft = _import_peft()
    except RuntimeError:
        return False
    peft_model_cls = getattr(peft, "PeftModel", None)
    return peft_model_cls is not None and isinstance(model, peft_model_cls)


def resolve_params_scope(trainer, params_scope: str) -> str:
    if params_scope is None:
        params_scope = PARAMS_SCOPE_AUTO
    params_scope = str(params_scope).lower()
    if params_scope not in VALID_PARAMS_SCOPES:
        raise ValueError("params_scope must be one of 'auto', 'model', or 'adapter'")

    model = unwrap_model(trainer)
    peft_model = is_peft_model(model)
    if params_scope == PARAMS_SCOPE_AUTO:
        return PARAMS_SCOPE_ADAPTER if peft_model else PARAMS_SCOPE_MODEL

    if params_scope == PARAMS_SCOPE_ADAPTER and not peft_model:
        raise ValueError("params_scope='adapter' requires trainer.model to be a PEFT PeftModel")

    if params_scope == PARAMS_SCOPE_MODEL and peft_model:
        logger.warning(
            "params_scope='model' on a PEFT model exchanges the full PEFT-wrapped state_dict keyspace; "
            "most PEFT jobs should use params_scope='auto' or 'adapter'."
        )
    return params_scope


def get_reference_state_dict(trainer, params_scope: str):
    params_scope = resolve_params_scope(trainer, params_scope)
    model = unwrap_model(trainer)
    if params_scope == PARAMS_SCOPE_ADAPTER:
        peft = _import_peft("PEFT adapter parameter validation")
        return dict(peft.get_peft_model_state_dict(model))
    return dict(model.state_dict())


def strip_server_key_prefix(params: Optional[Mapping], server_key_prefix: Optional[str]):
    if not params:
        return {}
    if not server_key_prefix:
        return dict(params)

    result = {}
    prefix_len = len(server_key_prefix)
    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError(f"parameter keys must be strings, got {type(key).__name__}.")
        new_key = key[prefix_len:] if key.startswith(server_key_prefix) else key
        if new_key in result:
            raise ValueError(f"server_key_prefix={server_key_prefix!r} produced duplicate parameter key {new_key!r}.")
        result[new_key] = value
    return result


def apply_server_key_prefix(params: Optional[Mapping], server_key_prefix: Optional[str]):
    if not params:
        return {}
    if not server_key_prefix:
        return dict(params)

    result = {}
    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError(f"parameter keys must be strings, got {type(key).__name__}.")
        new_key = f"{server_key_prefix}{key}"
        if new_key in result:
            raise ValueError(f"server_key_prefix={server_key_prefix!r} produced duplicate parameter key {new_key!r}.")
        result[new_key] = value
    return result


def _as_tensor_params(params: Mapping, reference: Mapping):
    torch = _import_torch()
    result = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            result[key] = value
            continue

        ref = reference.get(key)
        dtype = getattr(ref, "dtype", None)
        try:
            result[key] = torch.as_tensor(value, dtype=dtype)
        except TypeError:
            result[key] = torch.as_tensor(value)
    return result


def _validate_incoming_params(reference: Mapping, params: Mapping, strict: bool):
    from nvflare.app_opt.pt.utils import inspect_model_params

    params = dict(params or {})
    report = inspect_model_params(reference, params)
    if report.shape_mismatches:
        raise RuntimeError(report.format_shape_mismatch_error())
    if params and not report.matched_keys:
        raise RuntimeError(report.format_zero_match_error())
    if report.unexpected_keys:
        if strict:
            raise RuntimeError(report.format_unexpected_keys_error())
        logger.warning(report.format_unexpected_keys_warning())
        return {key: params[key] for key in report.matched_keys}, report
    return params, report


def validate_incoming_params(reference: Mapping, params: Mapping, strict: bool):
    params_to_load, _ = _validate_incoming_params(reference, params, strict)
    return params_to_load


def load_params(
    trainer,
    params: Optional[Mapping],
    params_scope: str = PARAMS_SCOPE_AUTO,
    strict: bool = True,
    server_key_prefix: Optional[str] = None,
):
    params = getattr(params, "params", params)
    params_scope = resolve_params_scope(trainer, params_scope)
    params = strip_server_key_prefix(params, server_key_prefix)
    if not params:
        return None

    reference = get_reference_state_dict(trainer, params_scope)
    params_to_load, report = _validate_incoming_params(reference, params, strict=strict)
    tensor_params = _as_tensor_params(params_to_load, reference)
    model = unwrap_model(trainer)

    if params_scope == PARAMS_SCOPE_ADAPTER:
        peft = _import_peft("loading PEFT adapter params")
        result = peft.set_peft_model_state_dict(model, tensor_params)
        _log_incompatible_keys(result)
        return report

    result = model.load_state_dict(tensor_params, strict=strict)
    _log_incompatible_keys(result)
    return report


def _log_incompatible_keys(load_result) -> None:
    if load_result is None:
        return

    missing_keys = getattr(load_result, "missing_keys", None)
    unexpected_keys = getattr(load_result, "unexpected_keys", None)
    if missing_keys is None and unexpected_keys is None:
        try:
            missing_keys, unexpected_keys = load_result
        except (TypeError, ValueError):
            missing_keys, unexpected_keys = [], []
    if missing_keys:
        logger.warning("Missing keys when loading global state_dict: %s", missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys when loading global state_dict: %s", unexpected_keys)


def extract_params(trainer, params_scope: str = PARAMS_SCOPE_AUTO):
    torch = _import_torch()
    params_scope = resolve_params_scope(trainer, params_scope)
    model = unwrap_model(trainer)
    if params_scope == PARAMS_SCOPE_ADAPTER:
        peft = _import_peft("extracting PEFT adapter params")
        state_dict = peft.get_peft_model_state_dict(model)
    else:
        accelerator = getattr(trainer, "accelerator", None)
        get_state_dict = getattr(accelerator, "get_state_dict", None)
        state_dict = get_state_dict(model) if callable(get_state_dict) else model.state_dict()

    result = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


def _exchange_format_value(exchange_format):
    value = getattr(exchange_format, "value", exchange_format)
    return str(value).lower() if value is not None else ExchangeFormat.NUMPY.value


def prepare_out_params(
    params: Optional[Mapping],
    exchange_format=ExchangeFormat.NUMPY,
    server_expected_format=None,
):
    torch = _import_torch()
    fmt = _exchange_format_value(exchange_format)
    server_fmt = _exchange_format_value(server_expected_format) if server_expected_format is not None else fmt
    as_numpy = fmt == ExchangeFormat.NUMPY.value
    cast_for_numpy_server = server_fmt == ExchangeFormat.NUMPY.value
    result = {}
    params = dict(params or {})
    for key, value in params.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue

        tensor = value.detach().cpu()
        if (as_numpy or cast_for_numpy_server) and tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        if as_numpy:
            result[key] = tensor.numpy()
        else:
            result[key] = tensor
    return result


def _positive_int(name: str, value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a positive integer.") from e
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _non_negative_int(name: str, value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a non-negative integer.") from e
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _get_arg_value(args, name: str, default=None):
    try:
        return getattr(args, name)
    except Exception:
        return default


def _world_size_from_args(args) -> int:
    world_size = _get_arg_value(args, "world_size", None)
    if world_size is None:
        world_size = os.environ.get("WORLD_SIZE", 1)
    try:
        world_size = int(world_size)
    except (TypeError, ValueError):
        world_size = 1
    return max(1, world_size)


def total_train_steps(dataset_len: int, args, total_rounds: int) -> int:
    total_rounds = _positive_int("total_rounds", total_rounds)

    max_steps = int(_get_arg_value(args, "max_steps", 0) or 0)
    if max_steps > 0:
        return max_steps * total_rounds

    dataset_len = _non_negative_int("dataset_len", dataset_len)
    num_train_epochs = float(_get_arg_value(args, "num_train_epochs", 1.0) or 0)
    if num_train_epochs < 0:
        raise ValueError("num_train_epochs must be non-negative.")

    per_device_batch_size = _positive_int(
        "per_device_train_batch_size", _get_arg_value(args, "per_device_train_batch_size", 1)
    )
    gradient_accumulation_steps = _positive_int(
        "gradient_accumulation_steps", _get_arg_value(args, "gradient_accumulation_steps", 1)
    )
    world_size = _world_size_from_args(args)

    batches_per_epoch = math.ceil(dataset_len / (per_device_batch_size * world_size)) if dataset_len else 0
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps) if batches_per_epoch else 0
    return math.ceil(num_train_epochs * steps_per_epoch) * total_rounds


def fl_exchange_dir(output_dir: str) -> str:
    return os.path.join(output_dir, FL_EXCHANGE_DIR)


def get_fl_exchange_dir(output_dir: str) -> str:
    return fl_exchange_dir(os.fspath(output_dir))


def find_checkpoint_for_step(output_dir: str, global_step: Optional[int]) -> Optional[str]:
    if global_step is None:
        return None
    path = os.path.join(output_dir, f"checkpoint-{int(global_step)}")
    return path if os.path.isdir(path) else None


def extract_params_from_checkpoint(checkpoint_dir: str, params_scope: str = PARAMS_SCOPE_MODEL) -> Optional[dict]:
    params_scope = str(params_scope or PARAMS_SCOPE_MODEL).lower()
    candidates = (
        ("adapter_model.safetensors", "adapter_model.safetensors.index.json", "adapter_model.bin")
        if params_scope == PARAMS_SCOPE_ADAPTER
        else ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )

    for file_name in candidates:
        path = os.path.join(checkpoint_dir, file_name)
        if not os.path.exists(path):
            continue
        if file_name.endswith(".index.json"):
            return _load_sharded_checkpoint(checkpoint_dir, path)
        return _load_checkpoint_file(path)

    return None


def _load_checkpoint_file(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return _detach_cpu_state_dict(load_file(path))

    torch = _import_torch()
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only is not available on older PyTorch. This fallback is only
        # for checkpoint files produced locally by HF Trainer or this adapter.
        state_dict = torch.load(path, map_location="cpu")
    return _detach_cpu_state_dict(state_dict)


def _load_sharded_checkpoint(checkpoint_dir: str, index_path: str) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid HuggingFace sharded checkpoint index: {index_path}")

    state_dict = {}
    for shard_name in dict.fromkeys(weight_map.values()):
        shard_path = os.path.join(checkpoint_dir, shard_name)
        if not os.path.exists(shard_path):
            raise RuntimeError(f"Missing HuggingFace checkpoint shard {shard_path} referenced by {index_path}")
        state_dict.update(_load_checkpoint_file(shard_path))
    return state_dict


def _detach_cpu_state_dict(state_dict: Mapping) -> dict:
    torch = _import_torch()
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state_dict.items()}


def params_nbytes(params: Optional[Mapping]) -> int:
    total = 0
    for value in dict(params or {}).values():
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
            continue

        nelement = getattr(value, "nelement", None)
        element_size = getattr(value, "element_size", None)
        if callable(nelement) and callable(element_size):
            total += int(nelement()) * int(element_size())
    return total


def write_params_exchange_file(output_dir: str, params: Mapping) -> dict:
    exchange_dir = fl_exchange_dir(os.fspath(output_dir))
    os.makedirs(exchange_dir, exist_ok=True)
    params = dict(params or {})

    try:
        return _write_safetensors_exchange_file(exchange_dir, params)
    except Exception as e:
        logger.debug("Falling back to torch params exchange file after safetensors write failure: %s", e)
        return _write_torch_exchange_file(exchange_dir, params)


def _write_safetensors_exchange_file(exchange_dir: str, params: Mapping) -> dict:
    torch = _import_torch()
    tensor_params = {}
    for key, value in params.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Cannot save non-tensor parameter {key!r} with safetensors.")
        tensor_params[key] = value.detach().cpu()

    from safetensors.torch import save_file

    path = _atomic_exchange_path(exchange_dir, ".safetensors")
    tmp_path = f"{path}.tmp"
    try:
        save_file(tensor_params, tmp_path)
        os.replace(tmp_path, path)
        return {"path": path, "format": PARAMS_EXCHANGE_FORMAT_SAFETENSORS}
    except Exception:
        _unlink_silent(tmp_path)
        raise


def _write_torch_exchange_file(exchange_dir: str, params: Mapping) -> dict:
    torch = _import_torch()
    path = _atomic_exchange_path(exchange_dir, ".pt")
    tmp_path = f"{path}.tmp"
    try:
        torch.save(_detach_cpu_state_dict(params), tmp_path)
        os.replace(tmp_path, path)
        return {"path": path, "format": PARAMS_EXCHANGE_FORMAT_TORCH}
    except Exception:
        _unlink_silent(tmp_path)
        raise


def _atomic_exchange_path(exchange_dir: str, suffix: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="params-", suffix=f"{suffix}.reserve", dir=exchange_dir)
    os.close(fd)
    os.unlink(tmp_path)
    return tmp_path[: -len(".reserve")]


def read_params_exchange_file(descriptor: Mapping) -> dict:
    path = os.fspath(descriptor["path"])
    fmt = str(descriptor.get("format") or "").lower()
    if fmt == PARAMS_EXCHANGE_FORMAT_SAFETENSORS:
        from safetensors.torch import load_file

        return _detach_cpu_state_dict(load_file(path))
    if fmt == PARAMS_EXCHANGE_FORMAT_TORCH:
        return _load_checkpoint_file(path)
    raise RuntimeError(f"Unsupported HF params exchange file format: {fmt!r}")


def cleanup_params_exchange_file(descriptor: Mapping) -> None:
    _unlink_silent(os.fspath(descriptor["path"]))


def _unlink_silent(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def write_params_to_checkpoint(trainer, checkpoint_dir: str, params: Mapping, params_scope: str, strict: bool = True):
    """Best-effort checkpoint-weight injection used by the explicit fallback strategy.

    Loading params into the trainer mutates the in-memory model before saving. The
    caller immediately resumes from the written checkpoint, so HF Trainer reloads
    the injected weights through its normal checkpoint path.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    load_params(trainer, params, params_scope=params_scope, strict=strict, server_key_prefix=None)
    model = unwrap_model(trainer)

    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(checkpoint_dir)
        return

    torch = _import_torch()
    state_dict = extract_params(trainer, params_scope)
    try:
        from safetensors.torch import save_file

        save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
    except Exception:
        torch.save(state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))
