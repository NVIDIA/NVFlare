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
"""Create an adapter-only NVFlare checkpoint for Nemotron 3 Nano LoRA PEFT."""

from __future__ import annotations

import argparse

import adapter_checkpoint

DEFAULT_MODEL_NAME_OR_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def define_parser():
    parser = argparse.ArgumentParser(description="Prepare an initial LoRA adapter checkpoint for NVFlare FedAvg.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--output", default="./models/nemotron3_nano_lora_init.pt")
    parser.add_argument(
        "--from_adapter_dir",
        default=None,
        help="Convert an existing Hugging Face PEFT adapter directory instead of instantiating the base model.",
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Device map passed to from_pretrained. Use "none" to disable.',
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading while materializing adapter shapes.",
    )
    return parser.parse_args()


def _split_target_modules(target_modules: str):
    if target_modules == "all-linear":
        return target_modules
    return [item.strip() for item in target_modules.split(",") if item.strip()]


def _create_adapter_state(args):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_split_target_modules(args.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return get_peft_model_state_dict(model), lora_config.to_dict()


def main():
    args = define_parser()
    if args.from_adapter_dir:
        state = adapter_checkpoint.load_adapter_state(args.from_adapter_dir)
        adapter_config = adapter_checkpoint.load_adapter_config(args.from_adapter_dir)
    else:
        state, adapter_config = _create_adapter_state(args)

    state = adapter_checkpoint.add_model_prefix(state)
    adapter_checkpoint.save_nvflare_adapter_checkpoint(state, args.output, adapter_config=adapter_config)
    print(f"Saved initial adapter checkpoint to {args.output}")


if __name__ == "__main__":
    main()
