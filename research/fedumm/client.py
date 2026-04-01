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

"""NVFlare FL client for federated BLIP-VQA fine-tuning."""

import argparse

import src.blip_backend  # noqa: F401 — triggers backend registration
import torch
from datasets import load_dataset
from src.common import (
    count_trainable_params,
    get_trainable_params,
    load_trainable_params,
    maybe_subsample,
    set_seed,
    shard_dataset,
    train_one_epoch,
)
from src.model_registry import get_backend
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import nvflare.client as flare


def parse_site_id(site_name: str) -> int:
    try:
        return int(site_name.split("-")[-1]) - 1
    except (ValueError, IndexError):
        return 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="", help="HF model id (uses backend default if empty).")
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_q_len", type=int, default=64)
    p.add_argument("--max_a_len", type=int, default=16)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--data_path", type=str, default="")
    p.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.0,
        help="Dirichlet concentration for non-IID partition. "
        "0 = IID round-robin, 0.1/0.5/1.0 = non-IID levels from paper.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    cache_dir = args.data_path or None

    backend = get_backend("blip_vqa")
    print(f">>> Backend: {backend.name}", flush=True)

    print(">>> Loading dataset ...", flush=True)
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

    flare.init()
    site = flare.get_site_name()
    writer = SummaryWriter()
    site_id = parse_site_id(site)
    train_hf = shard_dataset(train_hf, args.num_clients, site_id, alpha=args.dirichlet_alpha, seed=args.seed)
    eval_hf = shard_dataset(eval_hf, args.num_clients, site_id, alpha=0.0, seed=args.seed)  # eval always IID
    print(f"[{site}] train={len(train_hf)}, eval={len(eval_hf)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = backend.build_model_and_processor(
        args.model_name_or_path,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        device,
    )
    print(f"[{site}] {count_trainable_params(model)}", flush=True)

    train_ds = backend.build_dataset(train_hf, processor, args.max_q_len, args.max_a_len)
    eval_ds = backend.build_dataset(eval_hf, processor, args.max_q_len, args.max_a_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=backend.collate_fn,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=backend.collate_fn,
        pin_memory=True,
    )

    while flare.is_running():
        input_model = flare.receive()
        cur_round = getattr(input_model, "current_round", None) or 0

        if input_model and getattr(input_model, "params", None):
            load_trainable_params(model, input_model.params, device)

        # -- validate --
        if flare.is_evaluate():
            acc = backend.evaluate(model, eval_loader, processor, device)
            print(f"[{site}] validate round={cur_round} acc={acc:.4f}", flush=True)
            writer.add_scalar("val/acc", acc, cur_round)
            flare.send(
                flare.FLModel(
                    params=None,
                    metrics={"val_accuracy": float(acc), "n_eval": len(eval_ds)},
                    meta={"n_eval": len(eval_ds)},
                )
            )
            continue

        # -- train --
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        steps_per_epoch = len(train_loader)
        loss = 0.0
        for epoch in range(args.local_epochs):
            print(f"[{site}] round={cur_round} epoch={epoch + 1}/{args.local_epochs}", flush=True)
            global_step_offset = cur_round * args.local_epochs * steps_per_epoch + epoch * steps_per_epoch
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.grad_accum,
                backend,
                prefix=f"[{site}] round={cur_round}",
                writer=writer,
                global_step_offset=global_step_offset,
            )
        acc = backend.evaluate(model, eval_loader, processor, device)
        steps = args.local_epochs * steps_per_epoch
        print(f"[{site}] train round={cur_round} loss={loss:.4f} acc={acc:.4f}", flush=True)
        writer.add_scalar("train/acc", acc, cur_round)

        flare.send(
            flare.FLModel(
                params=get_trainable_params(model),
                metrics={
                    "train_loss": float(loss),
                    "local_acc": float(acc),
                    "n_train": len(train_ds),
                    "n_eval": len(eval_ds),
                },
                meta={"NUM_STEPS_CURRENT_ROUND": steps, "n_train": len(train_ds), "n_eval": len(eval_ds)},
            )
        )

    writer.close()


if __name__ == "__main__":
    main()
