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
"""Create an NVFlare full-model checkpoint for Nemotron 3 Nano SFT."""

from __future__ import annotations

import argparse

import model_checkpoint

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def define_parser():
    parser = argparse.ArgumentParser(description="Prepare an initial full-model checkpoint for NVFlare FedAvg SFT.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--output", default="./models/nemotron3_nano_4b_sft_init.pt")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Device map passed to from_pretrained. Use "none" to disable.',
    )
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    return parser.parse_args()


def _resolve_torch_dtype(torch_dtype: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[torch_dtype]


def main():
    import torch
    from transformers import AutoModelForCausalLM

    args = define_parser()
    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model_checkpoint.save_nvflare_model_checkpoint(state, args.output)
    total_bytes = sum(
        value.numel() * value.element_size() for value in state.values() if isinstance(value, torch.Tensor)
    )
    print(f"Saved initial full-model checkpoint to {args.output} ({total_bytes / (1024 ** 3):.2f} GiB)")


if __name__ == "__main__":
    main()
