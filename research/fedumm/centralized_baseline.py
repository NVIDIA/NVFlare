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

"""Centralized (non-FL) training baseline for BLIP-VQA."""

import argparse
import os

import src.blip_backend  # noqa: F401 — registers the blip_vqa backend
import torch
from datasets import load_dataset
from src.common import count_trainable_params, maybe_subsample, set_seed, train_one_epoch
from src.model_registry import get_backend
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="")
    p.add_argument("--output_dir", type=str, default="/tmp/nvflare/workspaces/fedumm/centralized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_q_len", type=int, default=64)
    p.add_argument("--max_a_len", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--data_path", type=str, default="")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))
    cache_dir = args.data_path or None
    backend = get_backend("blip_vqa")

    train_hf = load_dataset(
        backend.hf_dataset_name(), split=backend.hf_train_split(), cache_dir=cache_dir, trust_remote_code=True
    )
    eval_hf = load_dataset(
        backend.hf_dataset_name(), split=backend.hf_eval_split(), cache_dir=cache_dir, trust_remote_code=True
    )
    keep = set(backend.keep_columns())
    train_hf = train_hf.remove_columns([c for c in train_hf.column_names if c not in keep])
    eval_hf = eval_hf.remove_columns([c for c in eval_hf.column_names if c not in keep])
    train_hf = maybe_subsample(train_hf, args.max_train_samples, args.seed)
    eval_hf = maybe_subsample(eval_hf, args.max_eval_samples, args.seed)
    print(f"Train: {len(train_hf)}, Eval: {len(eval_hf)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = backend.build_model_and_processor(
        args.model_name_or_path, args.lora_r, args.lora_alpha, args.lora_dropout, device
    )
    print(count_trainable_params(model))

    train_ds = backend.build_dataset(train_hf, processor, args.max_q_len, args.max_a_len)
    eval_ds = backend.build_dataset(eval_hf, processor, args.max_q_len, args.max_a_len)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=backend.collate_fn,
        pin_memory=True,
    )
    eval_ld = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=backend.collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    steps_per_epoch = len(train_ld)
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}", flush=True)
        loss = train_one_epoch(
            model,
            train_ld,
            optimizer,
            device,
            args.grad_accum,
            backend,
            prefix=f"epoch={epoch + 1}",
            writer=writer,
            global_step_offset=epoch * steps_per_epoch,
        )
        acc = backend.evaluate(model, eval_ld, processor, device)
        writer.add_scalar("val/acc", acc, epoch + 1)
        print(f"Epoch {epoch + 1}/{args.num_epochs}  loss={loss:.4f}  acc={acc:.4f}")

    model.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
