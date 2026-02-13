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
"""ESM2 server module: loads checkpoint state_dict for NVFlare FedAvg (no Megatron/Lightning init)."""

import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch

from nvflare.fuel.utils.network_utils import get_open_ports


def _checkpoint_key_to_client(k: str) -> str:
    for old, new in (
        ("encoder.layers.self_attention.", "encoder.layers.0.self_attention."),
        ("encoder.layers.mlp.", "encoder.layers.0.mlp."),
    ):
        if old in k:
            k = k.replace(old, new, 1)
    return k


def _expand_checkpoint_state_dict(sd: OrderedDict) -> OrderedDict:
    """Split layer-stacked tensors [n, ...] into per-layer keys (layers.0.*, layers.1.*, ...)."""
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
    """Holds state_dict loaded from checkpoint; BioNeMoParamsFilter adds prefix when sending to client."""

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__()
        path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(path) and not os.path.isdir(path):
            raise FileNotFoundError(f"Checkpoint path does not exist or is not a file/directory: {checkpoint_path!r}")
        sd = load_state_dict_from_checkpoint_path(checkpoint_path)
        if sd is None:
            raise ValueError(
                f"Checkpoint is missing or invalid (could not load state dict from {checkpoint_path!r}). "
                "Ensure the path points to a valid NeMo or PyTorch checkpoint."
            )
        self._state_dict = _expand_checkpoint_state_dict(sd)

    @staticmethod
    def _stored_key(k: str) -> str:
        if k.startswith("module.module."):
            return k[len("module.") :]
        return k

    def state_dict(self, *args, **kwargs):
        return OrderedDict(self._state_dict)

    def load_state_dict(self, state_dict, strict: bool = True):
        self._state_dict = OrderedDict((self._stored_key(k), v) for k, v in state_dict.items())
        return None


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
    d = loaded
    for key in ("model", "state_dict", "weights", "checkpoint"):
        if key in loaded and isinstance(loaded[key], (dict, OrderedDict)):
            d = loaded[key]
            break
    if d is None:
        return None
    if all(isinstance(v, torch.Tensor) for v in d.values()):
        return OrderedDict(d)
    flat = _flatten_state_dict(d)
    return flat if flat and all(isinstance(v, torch.Tensor) for v in flat.values()) else None


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
        os.environ.setdefault("MASTER_PORT", str(get_open_ports(1)[0]))
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    ckpt_dir = os.path.abspath(weights_dir)
    try:
        loaded_sd = load_plain_tensors(ckpt_dir)
        if not isinstance(loaded_sd, dict):
            return None
        out = OrderedDict((k, v.cpu() if v.is_cuda else v) for k, v in loaded_sd.items() if isinstance(v, torch.Tensor))
        return out if out else None
    except Exception as e:
        warnings.warn(f"NeMo distributed checkpoint load failed: {e}", UserWarning, stacklevel=2)
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
        candidate = os.path.join(path, "weights", "common.pt")
        if os.path.isfile(candidate):
            try:
                loaded = torch.load(candidate, map_location="cpu", weights_only=False)
            except Exception:
                pass
    if loaded is None or not isinstance(loaded, dict):
        return None
    return _extract_state_dict(loaded)
