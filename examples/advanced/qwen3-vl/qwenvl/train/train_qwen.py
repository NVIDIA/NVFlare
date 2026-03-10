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
#
# Vendored/adapted from Qwen3-VL qwen-vl-finetune. See NOTICE and
# https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE

import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers

# Ensure example root (parent of qwenvl) is on path for qwenvl.data / qwenvl.train.argument imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Trainer,
)

from .trainer import replace_qwen2_vl_attention_class

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _is_rank0_or_single_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # For PEFT models, save adapter-only artifacts to reduce checkpoint I/O and
    # allow fast adapter reload paths.
    try:
        from peft import PeftModel

        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        if isinstance(model_to_save, PeftModel):
            if trainer.args.should_save:
                model_to_save.save_pretrained(output_dir)
            return
    except ImportError:
        pass

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad_(True)
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)


def _load_base_model_from_path(model_name_or_path, cache_dir, attn_implementation, bf16):
    """Load base (non-PEFT) model by path; used for full checkpoints and for base when loading adapter."""
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen3vl"
    elif model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen3vl"
    elif model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen2.5vl"
    elif model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen2vl"
    else:
        raise ValueError(f"Unsupported Qwen VL model_type: {model_type!r} for {model_name_or_path}")
    return model, model_type


def _to_cpu_state_dict(state_dict: dict) -> dict:
    cpu_state = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def _extract_fl_state_dict(model, lora_only: bool) -> dict:
    model_to_export = model.module if hasattr(model, "module") else model
    if lora_only:
        from peft import get_peft_model_state_dict

        return _to_cpu_state_dict(get_peft_model_state_dict(model_to_export))
    return _to_cpu_state_dict(model_to_export.state_dict())


def _maybe_add_default_adapter_name(key: str) -> str:
    for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        token = f".{marker}."
        default_token = f".{marker}.default."
        if token in key and default_token not in key:
            return key.replace(token, default_token, 1)
    return key


def _map_peft_state_dict_to_model_keys(model, state_dict: dict) -> tuple[dict, list]:
    model_keys = set(model.state_dict().keys())
    mapped = {}
    unmatched = []
    for key, value in state_dict.items():
        if key in model_keys:
            mapped[key] = value
            continue
        alt = _maybe_add_default_adapter_name(key)
        if alt in model_keys:
            mapped[alt] = value
        else:
            unmatched.append(key)
    return mapped, unmatched


def _load_initial_state_dict(model, state_dict: dict) -> None:
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    if PeftModel is not None and isinstance(model, PeftModel):
        mapped_state, unmatched = _map_peft_state_dict_to_model_keys(model, state_dict)
        if not mapped_state:
            raise RuntimeError(
                "No incoming LoRA keys matched the PEFT model state_dict; refusing to continue with stale local weights."
            )
        if unmatched:
            sample = ", ".join(unmatched[:3])
            raise RuntimeError(
                f"Failed to map {len(unmatched)}/{len(state_dict)} incoming LoRA keys. "
                f"Example unmatched keys: {sample}"
            )
        incompatible = model.load_state_dict(mapped_state, strict=False)
        rank0_print(
            "Loaded initial FL LoRA state dict with "
            f"{len(incompatible.missing_keys)} missing and {len(incompatible.unexpected_keys)} unexpected keys."
        )
        return

    incompatible = model.load_state_dict(state_dict, strict=False)
    rank0_print(
        "Loaded initial FL state dict with "
        f"{len(incompatible.missing_keys)} missing and {len(incompatible.unexpected_keys)} unexpected keys."
    )


def train(
    attn_implementation="flash_attention_2",
    initial_state_dict: Optional[dict] = None,
    return_state_dict: bool = False,
):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Support adapter-only dir (e.g. from NVFlare LoRA exchange): load base from config then load adapter
    adapter_config_path = pathlib.Path(model_args.model_name_or_path) / "adapter_config.json"
    model_loaded_as_peft = False
    if adapter_config_path.is_file():
        import json

        from peft import PeftModel

        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name_or_path = adapter_config.get("base_model_name_or_path", model_args.model_name_or_path)
        base_model, data_args.model_type = _load_base_model_from_path(
            base_model_name_or_path,
            training_args.cache_dir,
            attn_implementation,
            training_args.bf16,
        )
        base_model.config.use_cache = False
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path)
        model.train()
        # Ensure adapter params are trainable (same as Qwen3-VL --lora_enable: only LoRA is trainable)
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        model_loaded_as_peft = True
        rank0_print(f"Loaded base from {base_model_name_or_path} + adapter from {model_args.model_name_or_path}")
    else:
        model, data_args.model_type = _load_base_model_from_path(
            model_args.model_name_or_path,
            training_args.cache_dir,
            attn_implementation,
            training_args.bf16,
        )

    rank0_print(f"the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_loaded_as_peft:
        # Already a PeftModel from adapter dir; no need to wrap again
        if _is_rank0_or_single_process():
            model.print_trainable_parameters()
    elif training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        rank0_print("LoRA enabled")
        for p in model.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout if training_args.lora_dropout is not None else 0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen attention
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.train()
        if _is_rank0_or_single_process():
            model.print_trainable_parameters()
    else:
        set_model(model_args, model)

        if _is_rank0_or_single_process():
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    # Disable gradient checkpointing for PeftModel to avoid "element 0 of tensors does not require grad"
    from peft import PeftModel

    if isinstance(model, PeftModel) and training_args.gradient_checkpointing:
        rank0_print("Disabling gradient_checkpointing for PeftModel to avoid backward grad_fn errors.")
        training_args.gradient_checkpointing = False

    if initial_state_dict is not None:
        _load_initial_state_dict(model, initial_state_dict)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if not return_state_dict:
        trainer.save_state()

    model.config.use_cache = True

    if return_state_dict:
        return _extract_fl_state_dict(trainer.model, lora_only=training_args.lora_enable or model_loaded_as_peft)

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
