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
"""
Qwen3-VL model for federated SFT.
Used by job.py (initial_model) and client.py (local training).
Provides an nn.Module interface so the PT persistor can save/load state_dict.
"""

import glob
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration


def load_state_dict_from_checkpoint(checkpoint_dir: str) -> dict:
    """Load state_dict from a HuggingFace-style checkpoint dir without loading the full model.

    Reads .safetensors (or pytorch_model.bin) so the client can send weights back to the server
    quickly and return to flare.receive(), avoiding cell_pipe send timeouts.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    state_dict = {}

    # Prefer safetensors (faster, no pickle)
    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensor_files:
        try:
            from safetensors.torch import load_file

            for path in safetensor_files:
                state_dict.update(load_file(path, device="cpu"))
            return state_dict
        except Exception:
            pass

    # Fallback: pytorch_model.bin
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict

    raise FileNotFoundError(f"No .safetensors or pytorch_model.bin in {checkpoint_dir}")


def _get_qwen_vl_model_class(model_name_or_path: str):
    """Return the correct ForConditionalGeneration class for the model's config (qwen2_5_vl vs qwen3_vl)."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(config, "model_type", None) == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            pass
    return Qwen2_5_VLForConditionalGeneration


def load_qwen_vl_from_pretrained(model_name_or_path: str, **kwargs):
    """Load Qwen VL model from path or HF ID using the class that matches the checkpoint config."""
    model_cls = _get_qwen_vl_model_class(model_name_or_path)
    return model_cls.from_pretrained(model_name_or_path, **kwargs)


class Qwen3VLModel(nn.Module):
    """Qwen3-VL model wrapper for use as initial_model in FedAvgRecipe.

    Model configuration uses the HuggingFace model ID (e.g. Qwen/Qwen3-VL-2B-Instruct).
    Supports both Qwen2.5-VL and Qwen3-VL checkpoints; the correct class is chosen from config.
    Use a Qwen3-VL model when training with the Qwen3-VL repo's train_qwen.py.
    """

    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct", **kwargs):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.model = load_qwen_vl_from_pretrained(model_name_or_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
