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
"""Create an adapter-only NVFlare checkpoint for Nemotron 3 LoRA PEFT."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace

import adapter_checkpoint
import model_profiles


def define_parser():
    parser = argparse.ArgumentParser(description="Prepare an initial LoRA adapter checkpoint for NVFlare FedAvg.")
    model_profiles.add_model_profile_argument(parser)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--from_adapter_dir",
        default=None,
        help="Convert an existing Hugging Face PEFT adapter directory instead of instantiating the base model.",
    )
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", default=None)
    parser.add_argument("--exclude_modules", default=None)
    parser.add_argument("--use_triton_lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--cp_size", type=int, default=None)
    parser.add_argument("--ep_size", type=int, default=None)
    parser.add_argument("--activation_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--automodel_command", default="automodel")
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
    return model_profiles.resolve_model_profile(parser.parse_args())


def _split_target_modules(target_modules: str):
    if target_modules == "all-linear":
        return target_modules
    return [item.strip() for item in target_modules.split(",") if item.strip()]


def _create_adapter_state(args):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    model_kwargs = {
        "revision": args.model_revision,
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


def _initialization_native_args(args, train_file: str):
    values = vars(args).copy()
    values.update(
        train_file=train_file,
        validation_file=None,
        seq_length=32,
        limit_train_samples=1,
        limit_validation_samples=None,
        balance_train_labels=True,
        use_chat_template=False,
        learning_rate=5e-5,
        micro_batch_size=1,
        global_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
    )
    return SimpleNamespace(**values)


def _create_lightning_adapter_state(args):
    import automodel_peft_client
    import yaml

    with tempfile.TemporaryDirectory(prefix="nvflare_lightning35_init_") as temp_dir:
        train_file = os.path.join(temp_dir, "initialization_sample.jsonl")
        with open(train_file, "w") as f:
            f.write(json.dumps({"sentence": "The agreement is valid for four years .", "label": " neutral"}) + "\n")
        native_args = _initialization_native_args(args, train_file)
        output_root = os.path.join(temp_dir, "native_adapter")
        config = automodel_peft_client._default_automodel_config(
            native_args,
            os.path.join(temp_dir, "checkpoints"),
            incoming_adapter_dir="",
        )
        config["checkpoint"]["restore_from"] = None
        config_path = os.path.join(temp_dir, "initialize_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        env = os.environ.copy()
        example_dir = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (example_dir, env.get("PYTHONPATH"))))
        env["NVFLARE_INITIALIZE_ONLY"] = "1"
        env["NVFLARE_OUTPUT_ADAPTER_DIR"] = output_root
        env["NVFLARE_MODEL_PROFILE"] = args.model_profile
        subprocess.run([args.automodel_command, "--nproc-per-node=1", config_path], check=True, env=env)
        adapter_dir = os.path.join(output_root, "model")
        state = adapter_checkpoint.load_adapter_state(adapter_dir)
    adapter_config = {
        "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
        "dim": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "match_all_linear": args.target_modules == "all-linear" and not args.exclude_modules,
        "target_modules": [] if args.target_modules == "all-linear" else _split_target_modules(args.target_modules),
        "exclude_modules": _split_target_modules(args.exclude_modules),
        "use_triton": args.use_triton_lora,
    }
    return state, adapter_config


def main():
    args = define_parser()
    if args.output is None:
        args.output = args.initial_adapter_ckpt
    if args.from_adapter_dir:
        state = adapter_checkpoint.load_adapter_state(args.from_adapter_dir)
        adapter_config = adapter_checkpoint.load_adapter_config(args.from_adapter_dir)
        existing_manifest = adapter_checkpoint.load_adapter_manifest(args.from_adapter_dir)
    else:
        existing_manifest = None
        if model_profiles.is_lightning35(args):
            state, adapter_config = _create_lightning_adapter_state(args)
        else:
            state, adapter_config = _create_adapter_state(args)

    state = adapter_checkpoint.align_adapter_state_strict(state, state)
    if model_profiles.is_lightning35(args):
        state = {key: value.float() for key, value in state.items()}
    adapter_identity = model_profiles.adapter_identity(args)
    manifest = existing_manifest or adapter_checkpoint.build_adapter_manifest(state, identity=adapter_identity)
    adapter_checkpoint.validate_adapter_manifest(manifest, state, expected=adapter_identity)
    nvflare_state = adapter_checkpoint.add_model_prefix(state)
    adapter_checkpoint.save_nvflare_adapter_checkpoint(
        nvflare_state,
        args.output,
        adapter_config=adapter_config,
        adapter_manifest=manifest,
    )
    print(f"Saved initial adapter checkpoint to {args.output}")


if __name__ == "__main__":
    main()
