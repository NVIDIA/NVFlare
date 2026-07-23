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

import argparse
import inspect
import os
import random

from model import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MODEL_NAME,
)

import nvflare.client.hf as flare


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Qwen SFT with nvflare.client.hf")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen")
    parser.add_argument("--train_mode", choices=("sft", "peft"), default="peft")
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--stream_metrics", action="store_true")
    return parser.parse_args()


def setup_distributed():
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


def load_jsonl_dataset(path: str):
    import datasets

    return datasets.load_dataset("json", data_files=path, split="train")


def format_example(example):
    if example.get("text"):
        return example["text"]

    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def filtered_kwargs(callable_obj, values: dict) -> dict:
    params = inspect.signature(callable_obj).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return values
    return {name: value for name, value in values.items() if name in params}


def make_sft_config(args):
    import torch
    from trl import SFTConfig

    values = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.local_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "disable_tqdm": True,
        "report_to": [],
        "max_length": args.max_length,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "use_cpu": not torch.cuda.is_available(),
        "remove_unused_columns": False,
        "seed": 0,
        "data_seed": 0,
    }
    return SFTConfig(**filtered_kwargs(SFTConfig.__init__, values))


def make_model_and_peft_config(args, local_rank: int):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    if args.train_mode == "sft":
        return model, None

    from peft import LoraConfig, TaskType

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return model, peft_config


def main():
    args = define_parser()

    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    rank, _, local_rank = setup_distributed()
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_jsonl_dataset(args.train_data)
    eval_dataset = load_jsonl_dataset(args.eval_data)
    model, peft_config = make_model_and_peft_config(args, local_rank)

    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=format_example,
        peft_config=peft_config,
    )

    if args.train_mode == "peft":
        from peft import PeftModel

        if not isinstance(trainer.model, PeftModel):
            raise RuntimeError("PEFT mode is enabled, but SFTTrainer did not wrap the model as a PeftModel.")

    flare.patch(
        trainer,
        params_scope="auto",
        server_key_prefix=None if args.train_mode == "peft" else "model.",
        local_epochs=args.local_epochs,
        stream_metrics=args.stream_metrics,
    )

    while flare.is_running():
        metrics = trainer.evaluate()
        if rank == 0 and metrics and "eval_loss" in metrics:
            print(f"eval_loss={float(metrics['eval_loss'])}")
        trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
