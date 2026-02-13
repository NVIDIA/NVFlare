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
# WITHOUT WARRANTIES OR CONDITIONS FOR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
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


def _checkpoint_key_to_client(k: str) -> str:
    """Rewrite NeMo checkpoint key (single layer) to client format (layer 0)."""
    for old, new in (
        ("encoder.layers.self_attention.", "encoder.layers.0.self_attention."),
        ("encoder.layers.mlp.", "encoder.layers.0.mlp."),
    ):
        if old in k:
            k = k.replace(old, new, 1)
    return k


def _expand_checkpoint_state_dict(sd: OrderedDict) -> OrderedDict:
    """Expand NeMo checkpoint state dict so layer-stacked tensors become per-layer keys.

    NeMo saves some weights with a leading dimension (num_layers), e.g. shape [33, 1280, 1280].
    The client expects one key per layer with shape [1280, 1280]. We split such tensors and
    emit module.encoder.layers.0.*, module.encoder.layers.1.*, ...
    """
    out = OrderedDict()
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        # Keys that are layer-stacked: encoder.layers.self_attention.* or encoder.layers.mlp.*
        if "encoder.layers.self_attention." not in k and "encoder.layers.mlp." not in k:
            out[_checkpoint_key_to_client(k)] = v
            continue
        if v.ndim < 1:
            out[_checkpoint_key_to_client(k)] = v
            continue
        num_layers = v.shape[0]
        # Split into per-layer keys
        if "encoder.layers.self_attention." in k:
            base = k.replace("encoder.layers.self_attention.", "encoder.layers.{}.self_attention.", 1)
        else:
            base = k.replace("encoder.layers.mlp.", "encoder.layers.{}.mlp.", 1)
        for i in range(num_layers):
            out[base.format(i)] = v[i].clone()
    return out


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
        # Expand layer-stacked tensors (e.g. [33, 1280, 1280] -> 33 keys with [1280, 1280]) and
        # normalize key names to match client (Megatron) format.
        self._state_dict = _expand_checkpoint_state_dict(sd) if sd is not None else OrderedDict()
        # Do NOT add a prefix here: BioNeMoParamsFilter adds "module." (fp32) when sending to client.
        self._module = None  # no BionemoLightningModule on server; state comes from checkpoint

    @staticmethod
    def _stored_key(k: str) -> str:
        """Normalize key from client (module.module.*) to stored form (module.*) for consistency."""
        if k.startswith("module.module."):
            return k[len("module.") :]
        return k

    def state_dict(self, *args, **kwargs):
        # Return keys as stored; BioNeMoParamsFilter adds prefix when sending to client.
        return OrderedDict(self._state_dict)

    def load_state_dict(self, state_dict, strict: bool = True):
        self._state_dict = OrderedDict((self._stored_key(k), v) for k, v in state_dict.items())
        # Return type compatible with torch.nn.Module.load_state_dict
        incompat = getattr(torch.nn.modules.module, "_IncompatibleKeys", None)
        if incompat is not None:
            return incompat(missing_keys=[], unexpected_keys=[])
        return None


def _get_module_state_dict(module: torch.nn.Module, *args, **kwargs) -> OrderedDict:
    """Return state_dict from module, bypassing Lightning wrappers that return empty."""
    out = module.state_dict(*args, **kwargs)
    if out:
        return out
    for attr in ("model", "module", "net", "_model"):
        inner = getattr(module, attr, None)
        if isinstance(inner, torch.nn.Module):
            out = inner.state_dict(*args, **kwargs)
            if out:
                return out
    out = OrderedDict()
    for name, param in module.named_parameters():
        out[name] = param
    for name, buf in module.named_buffers():
        out[name] = buf
    if out:
        return out
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Module):
            child_sd = _get_module_state_dict(child, *args, **kwargs)
            if child_sd:
                return OrderedDict((f"{name}.{k}", v) for k, v in child_sd.items())
    return OrderedDict()


def _flatten_state_dict(d: dict, prefix: str = "") -> OrderedDict:
    out = OrderedDict()
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, torch.Tensor):
            out[key] = v
        elif isinstance(v, (dict, OrderedDict)):
            out.update(_flatten_state_dict(v, key))
    return out


def _extract_state_dict(loaded: dict) -> Optional[OrderedDict]:
    for key in ("model", "state_dict", "weights", "checkpoint"):
        if key in loaded and isinstance(loaded[key], (dict, OrderedDict)):
            d = loaded[key]
            break
    else:
        d = loaded
    if not d:
        return None
    if all(isinstance(v, torch.Tensor) for v in d.values()):
        return OrderedDict(d)
    flat = _flatten_state_dict(d)
    if flat and all(isinstance(v, torch.Tensor) for v in flat.values()):
        return flat
    return None


def _load_nemo_distributed_checkpoint(path: str) -> Optional[OrderedDict]:
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
        os.environ.setdefault("MASTER_PORT", str(12355 + (os.getpid() % 1000)))
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    ckpt_dir = os.path.abspath(weights_dir)
    try:
        loaded_sd = load_plain_tensors(ckpt_dir)
        if not isinstance(loaded_sd, dict) or len(loaded_sd) == 0:
            return None
        out = OrderedDict()
        for k, v in loaded_sd.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu() if v.is_cuda else v
        if out:
            return out
    except Exception as e:
        warnings.warn(f"NeMo distributed checkpoint load failed: {e}", UserWarning, stacklevel=2)
        return None
    return None


def load_state_dict_from_checkpoint_path(checkpoint_path: str) -> Optional[OrderedDict]:
    path = os.path.abspath(checkpoint_path)
    loaded = None
    if os.path.isfile(path):
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
    elif os.path.isdir(path):
        result = _load_nemo_distributed_checkpoint(path)
        if result is not None:
            return result
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


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        return self.weights * x
