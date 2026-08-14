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
from pathlib import Path

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
    parser.add_argument("--data_root", type=str, default="/tmp/nvflare/hello-huggingface/data")
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


def precision_config(torch):
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    fp16_enabled = cuda_available and not bf16_supported
    if bf16_supported:
        dtype = torch.bfloat16
    elif fp16_enabled:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return bf16_supported, fp16_enabled, dtype


def make_sft_config(output_dir: Path):
    import torch
    from trl import SFTConfig

    bf16_enabled, fp16_enabled, _ = precision_config(torch)
    values = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "lr_scheduler_type": "constant",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "disable_tqdm": True,
        "report_to": [],
        "max_length": 256,
        "bf16": bf16_enabled,
        "fp16": fp16_enabled,
        "use_cpu": not torch.cuda.is_available(),
        "remove_unused_columns": False,
        "seed": 0,
        "data_seed": 0,
    }
    return SFTConfig(**filtered_kwargs(SFTConfig.__init__, values))


def make_model_and_peft_config(model_name_or_path: str, local_rank: int):
    import torch
    from transformers import AutoModelForCausalLM

    _, _, dtype = precision_config(torch)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    from peft import LoraConfig, TaskType

    peft_config = LoraConfig(
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
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
    flare.init(rank=rank)
    site_name = flare.get_site_name()
    site_data_dir = Path(args.data_root).expanduser().resolve() / site_name
    train_data = site_data_dir / "train.jsonl"
    eval_data = site_data_dir / "valid.jsonl"
    if not train_data.is_file() or not eval_data.is_file():
        raise FileNotFoundError(f"Missing prepared data under {site_data_dir}. Run `python prepare_data.py` first.")
    output_dir = Path("outputs") / site_name

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_jsonl_dataset(str(train_data))
    eval_dataset = load_jsonl_dataset(str(eval_data))
    model, peft_config = make_model_and_peft_config(args.model_name_or_path, local_rank)

    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=format_example,
        peft_config=peft_config,
    )

    from peft import PeftModel

    if not isinstance(trainer.model, PeftModel):
        raise RuntimeError("SFTTrainer did not wrap the model as a PeftModel.")

    flare.patch(trainer)
    # Defaults are enough for this example. If needed, replace the line above with
    # optional settings such as:
    # flare.patch(trainer, local_epochs=1, stream_metrics=True)
    # flare.patch(trainer, local_steps=100)
    # flare.patch(trainer, params_scope="adapter", server_key_prefix="model.")

    while flare.is_running():
        metrics = trainer.evaluate()
        if rank == 0 and metrics and "eval_loss" in metrics:
            print(f"eval_loss={float(metrics['eval_loss'])}")
        trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
