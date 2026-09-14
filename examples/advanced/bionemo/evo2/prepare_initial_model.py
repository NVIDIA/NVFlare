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
"""Create the common initial Evo2 LoRA/classification-head checkpoint."""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import adapter_checkpoint
import evo2_runtime
import provenance
import torch


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bionemo", "mock"), default="bionemo")
    parser.add_argument("--base-checkpoint", default="./models/evo2_1b_bf16_mbridge")
    parser.add_argument("--data-file", default="./data/train/pooled.jsonl")
    parser.add_argument("--output", default="./models/evo2_lora_init.pt")
    parser.add_argument("--work-dir", default="/tmp/nvflare/evo2_initialize")
    parser.add_argument("--classifier-file", default=None)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--peft-mode", choices=("lora", "head-only"), default="lora")
    parser.add_argument("--lora-dim", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="linear_qkv,linear_proj,linear_fc1,linear_fc2,dense_projection,dense",
    )
    return parser


def create_mock_state(peft_mode: str, seed: int) -> OrderedDict[str, torch.Tensor]:
    """Create a tiny deterministic state for CPU workflow validation."""

    generator = torch.Generator().manual_seed(seed)
    state = OrderedDict()
    if peft_mode == "lora":
        state["decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"] = torch.randn(
            4, 8, generator=generator
        )
        state["decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"] = torch.zeros(8, 4)
    state["decoder.classification_head.weight"] = torch.randn(3, 8, generator=generator)
    state["decoder.classification_head.bias"] = torch.zeros(3)
    return state


def prepare_initial_model(args: argparse.Namespace) -> OrderedDict[str, torch.Tensor]:
    """Build and persist an identical initialization for every federated site."""

    training_inputs = {"data_file": None, "base_checkpoint": None, "classifier_file": None}
    if args.backend == "mock":
        state = create_mock_state(args.peft_mode, args.seed)
    else:
        missing = [path for path in (args.data_file,) if not os.path.isfile(path)]
        if not os.path.isdir(args.base_checkpoint):
            missing.append(args.base_checkpoint)
        if missing:
            raise FileNotFoundError("Missing BioNeMo initialization inputs:\n" + "\n".join(missing))
        classifier_file = evo2_runtime.resolve_classifier_path(args.classifier_file)
        training_inputs = {
            "data_file": provenance.jsonl_identity(args.data_file, label="initialization data"),
            "base_checkpoint": provenance.directory_identity(args.base_checkpoint),
            "classifier_file": provenance.file_identity(classifier_file),
        }
        state = evo2_runtime.initialize_trainable_state(
            classifier_path=args.classifier_file,
            base_checkpoint=str(Path(args.base_checkpoint).resolve()),
            data_file=str(Path(args.data_file).resolve()),
            result_dir=str(Path(args.work_dir).resolve()),
            seq_length=args.seq_length,
            seed=args.seed,
            peft_mode=args.peft_mode,
            lora_dim=args.lora_dim,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=evo2_runtime.parse_lora_targets(args.lora_target_modules),
        )

    metadata = {
        "backend": args.backend,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "exchange_dtype": adapter_checkpoint.EXCHANGE_DTYPE_NAME,
        "peft_mode": args.peft_mode,
        "seed": args.seed,
        "seq_length": args.seq_length,
        "lora_dim": args.lora_dim if args.peft_mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.peft_mode == "lora" else None,
        "lora_dropout": args.lora_dropout if args.peft_mode == "lora" else None,
        "lora_target_modules": (
            list(evo2_runtime.parse_lora_targets(args.lora_target_modules)) if args.peft_mode == "lora" else []
        ),
        "training_inputs": training_inputs,
    }
    adapter_checkpoint.save_nvflare_checkpoint(state, args.output, metadata=metadata)
    print(
        f"Saved {len(state)} trainable tensors ({adapter_checkpoint.state_dict_size_mb(state):.2f} MiB) "
        f"to {Path(args.output).resolve()}"
    )
    return state


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    prepare_initial_model(args)


if __name__ == "__main__":
    main()
