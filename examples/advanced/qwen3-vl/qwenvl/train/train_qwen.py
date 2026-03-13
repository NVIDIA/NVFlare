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

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers

# Ensure example root (parent of qwenvl) is on path for qwenvl.data / qwenvl.train.argument imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    load_state_dict_from_checkpoint,
    map_adapter_state_dict_for_peft_model,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from transformers import AutoConfig, AutoProcessor

from .trainer import QwenTrainer
from .trainer import print_trainable_parameters as print_qwen_text_trainable_parameters
from .trainer import print_trainable_parameters_visual as print_qwen_visual_trainable_parameters
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
    visual_module = _get_qwen_vl_visual_module(model)
    text_module = _get_qwen_vl_text_module(model)
    lm_head = _get_qwen_vl_lm_head(model)

    if model_args.tune_mm_vision:
        for n, p in visual_module.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in visual_module.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in text_module.named_parameters():
            p.requires_grad = True
        lm_head.requires_grad_(True)
    else:
        for n, p in text_module.named_parameters():
            p.requires_grad = False
        lm_head.requires_grad_(False)


def _get_qwen_vl_visual_module(model):
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "visual"):
            return candidate.visual
    raise AttributeError(f"Expected Qwen VL model with a visual tower, got {model.__class__.__name__}")


def _get_qwen_vl_text_module(model):
    candidates = (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "embed_tokens") and hasattr(candidate, "layers"):
            return candidate
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and hasattr(language_model, "embed_tokens") and hasattr(language_model, "layers"):
            return language_model
    raise AttributeError(f"Expected Qwen VL text backbone with embed_tokens/layers, got {model.__class__.__name__}")


def _get_qwen_vl_lm_head(model):
    for candidate in (model, getattr(model, "model", None), _get_qwen_vl_text_module(model)):
        if candidate is not None and hasattr(candidate, "lm_head"):
            return candidate.lm_head
    raise AttributeError(f"Expected Qwen VL model with lm_head, got {model.__class__.__name__}")


def _load_base_model_from_path(model_name_or_path, cache_dir, attn_implementation, bf16):
    """Load base (non-PEFT) model by path; used for full checkpoints and for base when loading adapter."""
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl_moe":
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
        except ImportError as e:
            raise ValueError(
                "Installed transformers package does not provide Qwen3-VL-MoE support. "
                f"Cannot load model_type={model_type!r} from {model_name_or_path}."
            ) from e

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen3vl"
    elif model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ValueError(
                "Installed transformers package does not provide Qwen3-VL support. "
                f"Cannot load model_type={model_type!r} from {model_name_or_path}."
            ) from e

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen3vl"
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        model_type = "qwen2.5vl"
    elif model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

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


def _load_initial_state_dict(model, state_dict: dict) -> None:
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    if PeftModel is not None and isinstance(model, PeftModel):
        mapped_state, unmatched = map_adapter_state_dict_for_peft_model(model, state_dict)
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

    model_keys = set(model.state_dict().keys())
    matched_keys = model_keys & set(state_dict.keys())
    if not matched_keys:
        sample = ", ".join(list(state_dict.keys())[:3])
        raise RuntimeError(
            "No incoming FL keys matched the freshly loaded model state_dict; refusing to continue with stale local "
            f"weights. Example incoming keys: {sample}"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    rank0_print(
        "Loaded initial FL state dict with "
        f"{len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected, "
        f"and {len(matched_keys)} matched keys."
    )


def _load_processor_with_fallback(primary_path: str, fallback_path: Optional[str] = None):
    try:
        return AutoProcessor.from_pretrained(primary_path, trust_remote_code=True)
    except OSError:
        if fallback_path and fallback_path != primary_path:
            rank0_print(
                f"Processor artifacts not found in {primary_path}; falling back to base model processor from {fallback_path}."
            )
            return AutoProcessor.from_pretrained(fallback_path, trust_remote_code=True)
        raise


def _load_tokenizer_with_fallback(primary_path: str, cache_dir: Optional[str], model_max_length: int):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "model_max_length": model_max_length,
        "padding_side": "right",
        "trust_remote_code": True,
        "use_fast": False,
    }
    try:
        return transformers.AutoTokenizer.from_pretrained(primary_path, **tokenizer_kwargs)
    except OSError:
        adapter_config_path = Path(primary_path) / "adapter_config.json"
        if adapter_config_path.is_file():
            import json

            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            fallback_path = adapter_config.get("base_model_name_or_path")
            if fallback_path and fallback_path != primary_path:
                rank0_print(
                    f"Tokenizer artifacts not found in {primary_path}; falling back to base model tokenizer from {fallback_path}."
                )
                return transformers.AutoTokenizer.from_pretrained(fallback_path, **tokenizer_kwargs)
        raise


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
    processor_source = model_args.model_name_or_path

    # Support adapter-only dir (e.g. from NVFlare LoRA exchange): load base from config then load adapter
    adapter_config_path = Path(model_args.model_name_or_path) / "adapter_config.json"
    model_loaded_as_peft = False
    if adapter_config_path.is_file():
        import json

        from peft import LoraConfig, TaskType, get_peft_model

        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name_or_path = adapter_config.get("base_model_name_or_path", model_args.model_name_or_path)
        processor_source = base_model_name_or_path
        base_model, data_args.model_type = _load_base_model_from_path(
            base_model_name_or_path,
            training_args.cache_dir,
            attn_implementation,
            training_args.bf16,
        )
        base_model.config.use_cache = False
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        # Build PeftModel on this process and load adapter weights with key mapping so keys match
        # regardless of single vs multi-GPU (avoids "missing adapter keys" when server/client key format differs)
        lora_config = LoraConfig(
            r=adapter_config.get("r", DEFAULT_LORA_R),
            lora_alpha=adapter_config.get("lora_alpha", DEFAULT_LORA_ALPHA),
            lora_dropout=adapter_config.get("lora_dropout", DEFAULT_LORA_DROPOUT),
            target_modules=adapter_config.get("target_modules", DEFAULT_LORA_TARGET_MODULES),
            bias=adapter_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        adapter_state = load_state_dict_from_checkpoint(str(model_args.model_name_or_path), lora_only=True)
        mapped_state, unmatched = map_adapter_state_dict_for_peft_model(model, adapter_state)
        if not mapped_state:
            raise RuntimeError(
                "No adapter keys could be mapped to the model. Adapter keys may not match this process "
                f"(e.g. multi-GPU). Unmatched sample: {unmatched[:3] if unmatched else 'none'}."
            )
        model.load_state_dict(mapped_state, strict=False)
        model.train()
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        n_trainable_lora = sum(1 for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad)
        if n_trainable_lora == 0:
            raise RuntimeError(
                "No LoRA parameters are trainable after loading adapter from " f"{model_args.model_name_or_path}."
            )
        model_loaded_as_peft = True
        rank0_print(
            f"Loaded base from {base_model_name_or_path} + adapter from {model_args.model_name_or_path} "
            f"({len(mapped_state)} keys mapped)."
        )
    else:
        model, data_args.model_type = _load_base_model_from_path(
            model_args.model_name_or_path,
            training_args.cache_dir,
            attn_implementation,
            training_args.bf16,
        )

    rank0_print(f"the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}")
    processor = _load_processor_with_fallback(model_args.model_name_or_path, processor_source)

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

    tokenizer = _load_tokenizer_with_fallback(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
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
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=DEFAULT_LORA_TARGET_MODULES,
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
            print_qwen_visual_trainable_parameters(_get_qwen_vl_visual_module(model))
            print_qwen_text_trainable_parameters(_get_qwen_vl_text_module(model))

    # Disable gradient checkpointing for PeftModel to avoid "element 0 of tensors does not require grad"
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    if (
        PeftModel is not None
        and isinstance(model, PeftModel)
        and training_args.gradient_checkpointing
        and attn_implementation == "flash_attention_2"
    ):
        rank0_print(
            "Disabling gradient_checkpointing for PeftModel with flash_attention_2 to avoid backward grad_fn errors."
        )
        training_args.gradient_checkpointing = False

    if initial_state_dict is not None:
        _load_initial_state_dict(model, initial_state_dict)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = QwenTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer.train()
    # Preserve Trainer state in both FL exchange modes; only the model artifact handling differs.
    trainer.save_state()

    model.config.use_cache = True

    if return_state_dict:
        rank0_print("Using in-memory FL exchange; returning trained state_dict without writing model checkpoints.")
        return _extract_fl_state_dict(trainer.model, lora_only=training_args.lora_enable or model_loaded_as_peft)

    rank0_print("Using checkpoint-based FL exchange; saving trained model artifacts to output_dir.")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
