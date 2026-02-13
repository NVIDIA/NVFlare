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
# WITHOUT WARRANTIES OR CONDITIONS FOR A PARTICULAR PURPOSE.  See the
# License for the specific language governing permissions and
# limitations under the License.
"""
Build ESM2 fine-tune module for NVFlare server-side initial model.

This module is used by job.py to instantiate the ESM2 architecture so that
FedAvgRecipe can load the initial checkpoint and aggregate client updates.
The same architecture (config + lightning module) as client.py is used so
state_dict keys match.

Use ESM2ModuleForServer via dict config in the recipe so the job config only
stores class path + args (no nn.Module instance). That avoids serialization
errors from callables/lambdas in the Lightning module object graph.
"""

import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

class ESM2ModuleForServer(torch.nn.Module):
    """ESM2 module for server: holds full state_dict loaded from checkpoint (no Megatron/Lightning init).

    Used via dict config so the job config never walks a real Lightning module. The server only needs
    state_dict() and load_state_dict(); we load the checkpoint with load_state_dict_from_checkpoint_path
    so the module has full content without instantiating BionemoLightningModule.
    """

    def __init__(
        self,
        checkpoint_path: str,
        task_type: str = "classification",
        encoder_frozen: bool = False,
        precision: str = "fp32",
        mlp_ft_dropout: float = 0.1,
        mlp_hidden_size: int = 256,
        mlp_target_size: int = 2,
        num_classes_for_metric: Optional[int] = None,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.task_type = task_type
        self.encoder_frozen = encoder_frozen
        self.precision = precision
        self.mlp_ft_dropout = mlp_ft_dropout
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_target_size = mlp_target_size
        self.num_classes_for_metric = num_classes_for_metric
        # Load full state dict from checkpoint so state_dict() returns real weights (no inner module needed)
        sd = load_state_dict_from_checkpoint_path(checkpoint_path)
        self._state_dict = OrderedDict(sd) if sd is not None else OrderedDict()
        self._module = None  # no BionemoLightningModule on server; state comes from checkpoint

    def state_dict(self, *args, **kwargs):
        return OrderedDict(self._state_dict)

    def load_state_dict(self, state_dict, strict: bool = True):
        self._state_dict = OrderedDict(state_dict)
        # Return type compatible with torch.nn.Module.load_state_dict
        incompat = getattr(torch.nn.modules.module, "_IncompatibleKeys", None)
        if incompat is not None:
            return incompat(missing_keys=[], unexpected_keys=[])
        return None


def _get_module_state_dict(module: torch.nn.Module, *args, **kwargs) -> OrderedDict:
    """Return state_dict from module, bypassing Lightning wrappers that return empty.

    NeMo/biobert Lightning modules may override state_dict() and return empty.
    Fall back to inner attributes, named_children, then build from named_parameters().
    """
    out = module.state_dict(*args, **kwargs)
    if out:
        return out
    # Try common inner attribute names (NeMo/Lightning/DDP)
    for attr in ("model", "module", "net", "_model"):
        inner = getattr(module, attr, None)
        if isinstance(inner, torch.nn.Module):
            out = inner.state_dict(*args, **kwargs)
            if out:
                return out
    # First named child with a non-empty state_dict
    for _name, child in module.named_children():
        if not isinstance(child, torch.nn.Module):
            continue
        out = child.state_dict(*args, **kwargs)
        if out:
            return out
    # Build from named_parameters()/named_buffers() (works when state_dict() is overridden to return {})
    out = OrderedDict()
    for name, param in module.named_parameters():
        out[name] = param
    for name, buf in module.named_buffers():
        out[name] = buf
    if out:
        return out
    # Recursively try first level of children with prefix (for nested wrappers that all return empty)
    for name, child in module.named_children():
        if not isinstance(child, torch.nn.Module):
            continue
        child_sd = _get_module_state_dict(child, *args, **kwargs)
        if child_sd:
            prefix = f"{name}."
            return OrderedDict((prefix + k, v) for k, v in child_sd.items())
    return OrderedDict()


def _flatten_state_dict(d: dict, prefix: str = "") -> OrderedDict:
    """Flatten nested dict of tensors to single-level state_dict (e.g. 'encoder.layer.0.weight' -> tensor)."""
    out = OrderedDict()
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, torch.Tensor):
            out[key] = v
        elif isinstance(v, (dict, OrderedDict)):
            out.update(_flatten_state_dict(v, key))
    return out


def _extract_state_dict(loaded: dict) -> Optional[OrderedDict]:
    """Extract a flat state_dict from a loaded checkpoint dict."""
    for key in ("model", "state_dict", "weights", "checkpoint"):
        if key in loaded and isinstance(loaded[key], (dict, OrderedDict)):
            d = loaded[key]
            break
    else:
        d = loaded
    if not d:
        return None
    # If already flat (all values are tensors), return as-is
    if all(isinstance(v, torch.Tensor) for v in d.values()):
        return OrderedDict(d)
    # Flatten nested dicts (e.g. NeMo/Megatron layout)
    flat = _flatten_state_dict(d)
    if flat and all(isinstance(v, torch.Tensor) for v in flat.values()):
        return flat
    return None


def _load_nemo_distributed_checkpoint(path: str) -> Optional[OrderedDict]:
    """Load state dict from NeMo distributed checkpoint (weights/ with metadata.json + .distcp).

    Requires torch.distributed to be initialized (e.g. world_size=1). Uses megatron.core.dist_checkpointing.
    Uses load_plain_tensors() which returns a flat StateDict of torch.Tensors (no model structure needed).
    """
    weights_dir = os.path.join(path, "weights")
    if not os.path.isdir(weights_dir):
        return None
    files = os.listdir(weights_dir)
    if "metadata.json" not in files or not any(f.endswith(".distcp") for f in files):
        return None
    try:
        from megatron.core.dist_checkpointing.serialization import load_plain_tensors
    except ImportError:
        try:
            from megatron.core import dist_checkpointing as dist_ckpt
            load_plain_tensors = getattr(dist_ckpt, "load_plain_tensors", None)
        except ImportError:
            load_plain_tensors = None
        if load_plain_tensors is None:
            return None
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        # Use a process-unique port to avoid EADDRINUSE when running alongside client processes.
        os.environ.setdefault("MASTER_PORT", str(12355 + (os.getpid() % 1000)))
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    ckpt_dir = os.path.abspath(weights_dir)
    try:
        # load_plain_tensors loads all sharded tensors as plain tensors; no model structure required
        loaded_sd = load_plain_tensors(ckpt_dir)
        if not isinstance(loaded_sd, dict) or len(loaded_sd) == 0:
            return None
        # Keep only tensor values and move to CPU for saving initial_global_model.pt
        out = OrderedDict()
        for k, v in loaded_sd.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu() if v.is_cuda else v
        if out:
            return out
    except Exception as e:
        warnings.warn(
            f"NeMo distributed checkpoint load failed: {e}",
            UserWarning,
            stacklevel=2,
        )
        return None
    return None


def load_state_dict_from_checkpoint_path(checkpoint_path: str) -> Optional[OrderedDict]:
    """Load a state_dict from the ESM2/NeMo checkpoint at checkpoint_path (file or directory).

    Use this to build the initial global model for the server without instantiating the
    full Lightning/Megatron model (which requires process group init). The checkpoint is
    the same one used by clients (restore-from-checkpoint-path).

    Supports: (1) plain .pt/.pth/.bin/.safetensors; (2) NeMo distributed checkpoint
    (directory with weights/common.pt, weights/metadata.json, weights/*.distcp).

    Returns:
        State dict suitable for torch.save(..., initial_global_model.pt), or None if loading fails.
    """
    path = os.path.abspath(checkpoint_path)
    loaded = None
    if os.path.isfile(path):
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
    elif os.path.isdir(path):
        # NeMo distributed checkpoint: weights in .distcp shards, metadata in weights/
        result = _load_nemo_distributed_checkpoint(path)
        if result is not None:
            return result
        # Plain checkpoint files
        candidates = [
            os.path.join(path, "weights", "common.pt"),
            os.path.join(path, "model_weights.pt"),
            os.path.join(path, "model.pt"),
            os.path.join(path, "weights.pt"),
            os.path.join(path, "pytorch_model.bin"),
            os.path.join(path, "model.safetensors"),
        ]
        for root, _dirs, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".pt", ".pth", ".bin", ".safetensors")):
                    candidates.append(os.path.join(root, f))
        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            try:
                if candidate.endswith(".safetensors") and load_safetensors is not None:
                    loaded = load_safetensors(candidate, device="cpu")
                else:
                    loaded = torch.load(candidate, map_location="cpu", weights_only=False)
                if loaded is not None:
                    break
            except Exception:
                continue
        if loaded is None:
            weights_dir = os.path.join(path, "weights")
            if os.path.isdir(weights_dir):
                merged = {}
                for f in sorted(os.listdir(weights_dir)):
                    if not f.endswith((".pt", ".pth")):
                        continue
                    try:
                        part = torch.load(os.path.join(weights_dir, f), map_location="cpu", weights_only=False)
                        if isinstance(part, dict):
                            for k, v in part.items():
                                if k in ("model", "state_dict") and isinstance(v, dict):
                                    merged.update(v)
                                else:
                                    merged[k] = v
                    except Exception:
                        continue
                if merged:
                    loaded = merged
    if loaded is None:
        return None
    if isinstance(loaded, dict):
        return _extract_state_dict(loaded)
    if hasattr(loaded, "state_dict") and callable(loaded.state_dict):
        try:
            return OrderedDict(loaded.state_dict())
        except Exception:
            pass
    return None


# def get_esm2_module(
#     checkpoint_path: str,
#     task_type: str = "classification",
#     encoder_frozen: bool = False,
#     precision: str = "fp32",
#     mlp_ft_dropout: float = 0.1,
#     mlp_hidden_size: int = 256,
#     mlp_target_size: int = 2,
#     scale_lr_layer: Optional[str] = None,
#     lr_multiplier: float = 1.0,
#     num_classes_for_metric: Optional[int] = None,
# ) -> torch.nn.Module:
#     """Build ESM2 fine-tune Lightning module for server-side initial model.

#     Uses the same config and builder as client.py so state_dict structure
#     matches. The returned module is an nn.Module (Lightning module) suitable
#     for FedAvgRecipe(model=..., initial_ckpt=...).

#     Args:
#         checkpoint_path: Path to the ESM2 checkpoint directory (e.g. from bionemo load()).
#         task_type: "classification" or "regression".
#         encoder_frozen: Whether to freeze the encoder.
#         precision: "fp32" or "bf16"/"fp16".
#         mlp_ft_dropout: Dropout for the MLP head.
#         mlp_hidden_size: Hidden size of the MLP head.
#         mlp_target_size: Output size (e.g. number of classes for classification).
#         scale_lr_layer: Layer name prefix for LR scaling (optional).
#         lr_multiplier: LR multiplier for scale_lr_layer (optional).
#         num_classes_for_metric: Number of classes for Accuracy metric (classification only).

#     Returns:
#         The ESM2 Lightning module (nn.Module).
#     """
#     tokenizer = get_tokenizer()

#     train_metric = None
#     if task_type == "regression":
#         valid_metric = TorchmetricsConfig(
#             class_path="MeanSquaredError", task="regression", metric_name="val_mse"
#         )
#     else:
#         num_classes = num_classes_for_metric if num_classes_for_metric is not None else mlp_target_size
#         valid_metric = TorchmetricsConfig(
#             class_path="Accuracy",
#             task="classification",
#             kwargs={
#                 "task": "multiclass",
#                 "threshold": 0.5,
#                 "num_classes": num_classes,
#             },
#             metric_name="val_acc",
#         )

#     # Mirror client.py: create config then set task-dependent attrs (same as train_model in client.py)
#     config: BioBertConfig = ESM2FineTuneSeqConfig(
#         task_type=task_type,
#         encoder_frozen=encoder_frozen,
#         params_dtype=get_autocast_dtype(precision),
#         pipeline_dtype=get_autocast_dtype(precision),
#         autocast_dtype=get_autocast_dtype(precision),
#         tensor_model_parallel_size=1,
#         pipeline_model_parallel_size=1,
#         initial_ckpt_path=str(checkpoint_path),
#         initial_ckpt_skip_keys_with_these_prefixes=[f"{task_type}_head"],
#         train_metric=train_metric,
#         valid_metric=valid_metric,
#     )
#     task_dependent_attr = {
#         "mlp_ft_dropout": mlp_ft_dropout,
#         "mlp_hidden_size": mlp_hidden_size,
#         "mlp_target_size": mlp_target_size,
#         "cnn_dropout": 0.25,
#         "cnn_hidden_size": 32,
#         "cnn_num_classes": 3,
#     }
#     for attr, value in task_dependent_attr.items():
#         if hasattr(config, attr):
#             setattr(config, attr, value)

#     optimizer = MegatronOptimizerModule(
#         config=OptimizerConfig(
#             lr=1e-4,
#             optimizer="adam",
#             use_distributed_optimizer=True,
#             weight_decay=0.01,
#             adam_beta1=0.9,
#             adam_beta2=0.98,
#         ),
#     )
#     # Do not set optimizer.scale_lr_cond to a lambda here: the server model is serialized into the
#     # job config, and lambdas cannot be serialized (inspect.getsourcefile fails on them).

#     module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer)
#     _materialize_lazy_module(module, tokenizer, task_type, mlp_target_size)
#     return module


# def _init_single_process_group_if_needed():
#     """Initialize a single-process torch.distributed group so configure_model() can run (e.g. BionemoLightningModule)."""
#     if torch.distributed.is_initialized():
#         return
#     import os
#     os.environ.setdefault("MASTER_ADDR", "localhost")
#     os.environ.setdefault("MASTER_PORT", "12355")
#     torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


# def _materialize_lazy_module(
#     module: torch.nn.Module,
#     tokenizer,
#     task_type: str,
#     mlp_target_size: int,
#     max_length: int = 32,
# ) -> None:
#     """Run one dummy forward so lazy-initialized parameters (e.g. biobert) are created.

#     BionemoLightningModule has 0 parameters until setup/configure_model run.
#     configure_model() requires the default process group; we init a single-process group if needed.
#     """
#     import warnings

#     # Trigger Lightning model creation (BionemoLightningModule creates model in setup/configure_model)
#     if callable(getattr(module, "setup", None)):
#         try:
#             module.setup("fit")
#         except Exception as e:
#             warnings.warn(f"module.setup('fit') failed: {e}", UserWarning, stacklevel=2)
#     if callable(getattr(module, "configure_model", None)):
#         try:
#             _init_single_process_group_if_needed()
#             module.configure_model()
#         except Exception as e:
#             warnings.warn(f"module.configure_model() failed: {e}", UserWarning, stacklevel=2)

#     batch = _make_dummy_batch(tokenizer, task_type, max_length)
#     if batch is None:
#         warnings.warn("Could not build dummy batch for materialization (state_dict may be empty)", UserWarning, stacklevel=2)
#         return
#     try:
#         module.eval()
#         with torch.no_grad():
#             if callable(getattr(module, "forward", None)):
#                 _ = module(**batch)
#             elif callable(getattr(module, "training_step", None)):
#                 _ = module.training_step(batch, batch_idx=0)
#             else:
#                 _ = module(batch)
#     except Exception as e:
#         warnings.warn(
#             f"Could not materialize lazy module (state_dict may be empty): {e}",
#             UserWarning,
#             stacklevel=2,
#         )


# def _make_dummy_batch(tokenizer, task_type: str, max_length: int):
#     """Build a minimal batch for a dummy forward. Tries HF-style then tokenize()-style."""
#     dummy_seq = "M"
#     batch = None
#     # Try 1: HuggingFace-style tokenizer(...) with return_tensors="pt"
#     try:
#         out = tokenizer(
#             [dummy_seq],
#             padding="max_length",
#             max_length=max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         if hasattr(out, "items"):
#             batch = {k: v for k, v in out.items()}
#         else:
#             batch = {"input_ids": out if isinstance(out, torch.Tensor) else torch.tensor([out])}
#     except (TypeError, AttributeError):
#         pass
#     # Try 2: tokenizer.tokenize() returns list of ids; wrap in tensor
#     if batch is None and hasattr(tokenizer, "tokenize"):
#         try:
#             ids = tokenizer.tokenize(dummy_seq)
#             if isinstance(ids, list):
#                 ids = ids[:max_length]
#                 if not ids:
#                     ids = [0]
#                 input_ids = torch.tensor([ids], dtype=torch.long)
#             else:
#                 input_ids = torch.tensor([[ids]], dtype=torch.long)
#             batch = {"input_ids": input_ids}
#         except Exception:
#             pass
#     # Try 3: tokenizer.encode() or single sequence
#     if batch is None and hasattr(tokenizer, "encode"):
#         try:
#             ids = tokenizer.encode(dummy_seq)
#             ids = (ids if isinstance(ids, list) else [ids])[:max_length] or [0]
#             batch = {"input_ids": torch.tensor([ids], dtype=torch.long)}
#         except Exception:
#             pass
#     if batch is None:
#         return None
#     device = batch["input_ids"].device
#     if task_type == "classification":
#         batch["labels"] = torch.zeros(1, dtype=torch.long, device=device)
#     else:
#         batch["labels"] = torch.zeros(1, dtype=torch.float32, device=device)
#     return batch
