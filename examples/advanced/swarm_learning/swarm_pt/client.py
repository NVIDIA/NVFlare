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

"""Swarm LoRA fine-tuning client.

Trains a Qwen2.5-1.5B causal-LM with LoRA adapters on the wikitext-2-raw-v1
dataset (Apache-2.0, no license acceptance required).  Only the LoRA adapter
parameters (~0.4 % of total weights) are exchanged each round, keeping
communication cost low regardless of base-model size.

Dataset heterogeneity: the training split is partitioned by site index so each
participant trains on a disjoint shard, simulating real-world data silos. A
shared validation split scores each received global adapter before local training.
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

# LoRA hyperparameters — must match model.py
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters
MAX_SEQ_LEN = 128
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
DEFAULT_LOCAL_STEPS = 10
DEFAULT_VALIDATION_STEPS = 10


def build_lora_model(model_path: str):
    """Load the base model and attach LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def build_dataloader(
    tokenizer,
    site_name: str,
    data_dir: str = None,
    n_shards: int = 4,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    split: str = "train",
) -> DataLoader:
    """Build a DataLoader for this site's training shard or the validation split.

    If data_dir is given, loads the Arrow dataset written by prepare_data.py.
    Otherwise falls back to the matching in-memory wikitext-2 split. Only the
    training split is sharded; validation is shared so every site scores the
    same received global model on the same examples.
    """
    if split not in ("train", "validation"):
        raise ValueError(f"split must be 'train' or 'validation', but got {split!r}")

    if data_dir is not None:
        from datasets import load_from_disk

        dataset_path = os.path.join(data_dir, site_name, "train") if split == "train" else os.path.join(data_dir, split)
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(
                f"Prepared {split} data not found at '{dataset_path}'. "
                f"Run prepare_data.py --n_clients <N> --output_dir {data_dir} first."
            )
        print(f"[{site_name}] Loading prepared {split} data from {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        print(f"[{site_name}] No --data_dir given; loading in-memory wikitext-2 {split} split")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
        if split == "train":
            try:
                site_idx = int(site_name.rsplit("-", 1)[-1]) - 1
            except ValueError:
                site_idx = 0
            dataset = dataset.shard(num_shards=n_shards, index=site_idx % n_shards)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", drop_last=True)


def apply_global_adapter(model, global_params: dict):
    """Overwrite the local LoRA adapter weights with those from the aggregator."""
    adapter_state = {}
    for k, v in global_params.items():
        if isinstance(v, np.ndarray):
            adapter_state[k] = torch.from_numpy(v).float()
        elif isinstance(v, torch.Tensor):
            adapter_state[k] = v.float()
        else:
            adapter_state[k] = torch.as_tensor(v, dtype=torch.float32)
    set_peft_model_state_dict(model, adapter_state)


def local_train(model, dataloader: DataLoader, steps: int) -> dict:
    """
    Fine-tune for `steps` gradient steps.  Returns the LoRA adapter weight
    *diff* (after − before) as a numpy dict for federated aggregation.
    """
    weights_before = {k: v.clone().detach() for k, v in get_peft_model_state_dict(model).items()}

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    device = next(model.parameters()).device
    model.train()
    data_iter = iter(dataloader)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 5 == 0 or step == steps - 1:
            print(f"  step={step + 1}/{steps}  loss={loss.item():.4f}", flush=True)

    weights_after = get_peft_model_state_dict(model)
    diff = {k: (weights_after[k] - weights_before[k]).float().detach().cpu() for k in weights_before}
    return diff


def evaluate_loss(model, dataloader: DataLoader, steps: int) -> float:
    """Evaluate the received global adapter before local training."""
    device = next(model.parameters()).device
    losses = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= steps:
                break
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            losses.append(loss.item())

    if not losses:
        raise ValueError("validation dataloader produced no complete batches")
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace Hub model ID or local path to the Qwen2.5 model",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Root directory of pre-split data written by prepare_data.py "
        "(e.g. /tmp/swarm_data). If omitted, falls back to in-memory shard.",
    )
    parser.add_argument("--local_steps", type=int, default=DEFAULT_LOCAL_STEPS)
    parser.add_argument("--validation_steps", type=int, default=DEFAULT_VALIDATION_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument(
        "--n_shards",
        type=int,
        default=4,
        help="Number of shards for in-memory data splitting (ignored when --data_dir is given). "
        "Should match --n_clients.",
    )
    args = parser.parse_args()

    flare.init()
    site_name = flare.get_site_name()
    print(f"[{site_name}] Loading model from '{args.model_path}'")

    model, tokenizer = build_lora_model(args.model_path)
    train_dataloader = build_dataloader(
        tokenizer,
        site_name,
        data_dir=args.data_dir,
        n_shards=args.n_shards,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    validation_dataloader = build_dataloader(
        tokenizer,
        site_name,
        data_dir=args.data_dir,
        n_shards=args.n_shards,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        split="validation",
    )

    print(
        f"[{site_name}] Datasets ready "
        f"({len(train_dataloader)} training batches, {len(validation_dataloader)} validation batches)"
    )

    round_num = 0
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        print(f"[{site_name}] Round {round_num}: applying global LoRA adapter")
        apply_global_adapter(model, input_model.params)

        val_loss = evaluate_loss(model, validation_dataloader, args.validation_steps)
        print(f"[{site_name}] Round {round_num}: received global adapter val_loss={val_loss:.4f}")

        print(f"[{site_name}] Round {round_num}: local training for {args.local_steps} steps")
        diff = local_train(model, train_dataloader, args.local_steps)

        flare.send(
            flare.FLModel(
                params_type=ParamsType.DIFF,
                params=diff,
                metrics={"val_loss": val_loss},
                meta={"NUM_STEPS_CURRENT_ROUND": args.local_steps},
            )
        )
        print(f"[{site_name}] Round {round_num}: submitted LoRA adapter diff")
        round_num += 1


if __name__ == "__main__":
    main()
