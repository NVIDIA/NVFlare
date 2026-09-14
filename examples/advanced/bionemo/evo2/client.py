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
"""NVFlare Client API entry point for federated Evo2 LoRA classification."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping

import evo2_adapter_checkpoint as adapter_checkpoint
import torch

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one federated Evo2 classification round.")
    parser.add_argument("--backend", choices=("bionemo", "mock"), default="bionemo")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--classifier-file", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-dim", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "dense_projection", "dense")),
    )
    parser.add_argument("--mock-delta", type=float, default=0.01)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.local_steps <= 0:
        raise ValueError("--local-steps must be positive.")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.eval_iters <= 0:
        raise ValueError("--eval-iters must be positive.")
    if args.backend == "bionemo":
        missing = [path for path in (args.train_file, args.validation_file) if not os.path.isfile(path)]
        if not os.path.isdir(args.base_checkpoint):
            missing.append(args.base_checkpoint)
        if missing:
            raise FileNotFoundError("Missing BioNeMo training inputs:\n" + "\n".join(missing))


def _mock_round(
    incoming_state: Mapping[str, torch.Tensor], delta: float
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    updated = OrderedDict()
    for name, value in incoming_state.items():
        if torch.is_floating_point(value):
            updated[name] = value.detach().cpu() + delta
        else:
            updated[name] = value.detach().cpu().clone()
    adapter_checkpoint.validate_trainable_state(updated, incoming_state, context="Mock trained state")
    return updated, {"validation_accuracy": max(0.0, min(1.0, 0.5 + delta)), "validation_ce_loss": 1.0}


def train_one_round(
    args: argparse.Namespace,
    incoming_state: Mapping[str, torch.Tensor],
    *,
    site_name: str,
    current_round: int,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float], str]:
    """Train from ``incoming_state`` and return an explicit DIFF update."""

    adapter_checkpoint.validate_trainable_state(incoming_state, incoming_state)
    os.makedirs(args.work_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=f"{site_name}_round_{current_round:03d}_", dir=args.work_dir)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.backend == "mock":
        mock_state, metrics = _mock_round(incoming_state, args.mock_delta)
        diff = adapter_checkpoint.compute_trainable_diff(mock_state, incoming_state)
    else:
        import evo2_runtime

        _updated_state, diff, metrics = evo2_runtime.train_round(
            incoming_state,
            classifier_path=args.classifier_file,
            base_checkpoint=args.base_checkpoint,
            train_file=args.train_file,
            validation_file=args.validation_file,
            result_dir=attempt_dir,
            local_steps=args.local_steps,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            eval_iters=args.eval_iters,
            seed=args.seed,
            round_index=current_round,
            lora_dim=args.lora_dim,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=evo2_runtime.parse_lora_targets(args.lora_target_modules),
        )

    adapter_checkpoint.validate_trainable_state(diff, incoming_state, context="Locally trained DIFF")
    metrics = dict(metrics)
    metrics.update(
        {
            "local_steps": float(args.local_steps),
            "runtime_seconds": time.monotonic() - started,
            "samples_available": float(args.sample_count),
            "received_mebibytes": adapter_checkpoint.state_dict_size_mb(incoming_state),
            "sent_mebibytes": adapter_checkpoint.state_dict_size_mb(diff),
            "peak_gpu_memory_mebibytes": (
                torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            ),
        }
    )
    return diff, metrics, attempt_dir


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    _validate_args(args)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))

    flare.init()
    input_model = flare.receive()
    if input_model is None:
        raise RuntimeError("NVFlare did not provide a global trainable state.")
    if input_model.params_type not in (None, flare.ParamsType.FULL):
        raise ValueError(f"Expected FULL global parameters, received {input_model.params_type}.")
    if (input_model.meta or {}).get("exchange_dtype") != adapter_checkpoint.EXCHANGE_DTYPE_NAME:
        raise ValueError(
            "NVFlare global model metadata must declare " f"exchange_dtype={adapter_checkpoint.EXCHANGE_DTYPE_NAME!r}."
        )

    site_name = flare.system_info().get("site_name", "unknown")
    current_round = input_model.current_round if input_model.current_round is not None else 0
    incoming_state = adapter_checkpoint.copy_trainable_state(
        input_model.params or {}, context="NVFlare global trainable state"
    )
    print(
        f"site={site_name}, round={current_round}, "
        f"received_trainable_mib={adapter_checkpoint.state_dict_size_mb(incoming_state):.2f}"
    )

    diff, metrics, attempt_dir = train_one_round(args, incoming_state, site_name=site_name, current_round=current_round)
    flare.send(
        flare.FLModel(
            params_type=flare.ParamsType.DIFF,
            params=diff,
            metrics=metrics,
            # Static recipe aggregation_weights contain each site's sample count.
            # A unit step here avoids multiplying those weights by optimizer steps.
            meta={
                FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1,
                "exchange_dtype": adapter_checkpoint.EXCHANGE_DTYPE_NAME,
            },
        )
    )
    print(
        f"site={site_name}, round={current_round}, sent_trainable_mib={metrics['sent_mebibytes']:.2f}, "
        f"attempt_dir={attempt_dir}"
    )
    if metrics.get("frozen_parameters_unchanged") == 1.0:
        print(f"site={site_name}, round={current_round}, frozen_backbone=verified")


if __name__ == "__main__":
    main()
