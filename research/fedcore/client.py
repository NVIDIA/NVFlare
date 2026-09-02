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

"""NVFlare client for valid-supervision FedCoRe training."""

import argparse
import json
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import LogitCompletionModel, effect_target
from src.features import load_cache_split
from src.federated import aggregation_meta, state_dict_for_update
from src.validation import non_negative_float, positive_float, positive_int, probability

import nvflare.client as flare

_TERMINATION_REQUESTED = False


def _tensor_state_dict(params: dict) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in params.items():
        clean = key[6:] if key.startswith("model.") else key
        result[clean] = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return result


def _train_round(
    model: LogitCompletionModel,
    payload: dict,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    task_weight: float,
    effect_weight: float,
    seed: int,
    current_round: int,
) -> dict[str, float]:
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paired_indices = payload["paired_mask"].nonzero(as_tuple=False).flatten()
    paired_count = int(paired_indices.numel())
    if paired_count == 0:
        return {"paired_examples": 0.0, "train_loss": 0.0, "task_loss": 0.0, "effect_loss": 0.0}

    features = payload["missing_features"][paired_indices].float()
    missing_logits = payload["missing_logits"][paired_indices].float()
    full_logits = payload["full_logits"][paired_indices].float()
    labels = payload["labels"][paired_indices].float()
    targets = effect_target(full_logits, missing_logits)
    round_seed = int(seed) + 1000 * int(current_round)
    loss_sum = task_sum = effect_sum = 0.0
    steps = 0
    model.train()

    # Keep sample order and dropout deterministic without mutating caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(round_seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
        generator = torch.Generator().manual_seed(round_seed)
        for _ in range(int(local_epochs)):
            order = torch.randperm(paired_count, generator=generator)
            for start in range(0, paired_count, int(batch_size)):
                indices = order[start : start + batch_size]
                predicted_delta = model(features[indices])
                completed = missing_logits[indices] + predicted_delta
                task_loss = F.binary_cross_entropy_with_logits(completed, labels[indices])
                delta_loss = F.smooth_l1_loss(predicted_delta, targets[indices])
                loss = float(task_weight) * task_loss + float(effect_weight) * delta_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("FedCoRe local training produced a non-finite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach().item())
                task_sum += float(task_loss.detach().item())
                effect_sum += float(delta_loss.detach().item())
                steps += 1
    return {
        "paired_examples": float(paired_count),
        "train_loss": loss_sum / max(1, steps),
        "task_loss": task_sum / max(1, steps),
        "effect_loss": effect_sum / max(1, steps),
    }


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FedCoRe classifier-logit completion operator.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=positive_int, default=128)
    parser.add_argument("--dropout", type=probability, default=0.1)
    parser.add_argument("--local-epochs", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--task-weight", type=non_negative_float, default=4.0)
    parser.add_argument("--effect-weight", type=non_negative_float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _shutdown_on_signal(_signum, _frame) -> None:
    """Request an orderly exit; the existing finally block owns teardown."""

    global _TERMINATION_REQUESTED
    _TERMINATION_REQUESTED = True


def main() -> None:
    global _TERMINATION_REQUESTED

    _TERMINATION_REQUESTED = False
    args = define_parser().parse_args()
    train_payload = load_cache_split(Path(args.cache_dir), args.site, "train")
    model = LogitCompletionModel(args.input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, seed=args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve() / args.site
    output_dir.mkdir(parents=True, exist_ok=True)

    flare.init()
    signal.signal(signal.SIGTERM, _shutdown_on_signal)
    try:
        while flare.is_running() and not _TERMINATION_REQUESTED:
            input_model = flare.receive()
            if input_model is None or _TERMINATION_REQUESTED:
                break
            current_round = int(getattr(input_model, "current_round", 0) or 0)
            model.load_state_dict(_tensor_state_dict(input_model.params), strict=True)
            start = time.perf_counter()
            train_metrics = _train_round(
                model=model,
                payload=train_payload,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                task_weight=args.task_weight,
                effect_weight=args.effect_weight,
                seed=args.seed,
                current_round=current_round,
            )
            paired_examples = int(train_metrics["paired_examples"])
            params = state_dict_for_update(model, paired_examples)
            round_metrics = {
                **train_metrics,
                "site": args.site,
                "round": current_round,
                "paired_examples": paired_examples,
                "sent_empty_update": not bool(params),
                "wall_time_seconds": time.perf_counter() - start,
            }
            with (output_dir / f"round_{current_round:03d}.json").open("w") as f:
                json.dump(round_metrics, f, indent=2, sort_keys=True)
                f.write("\n")
            output_model = flare.FLModel(
                params=params,
                metrics={
                    "train_loss": float(train_metrics["train_loss"]),
                    "paired_examples": float(paired_examples),
                },
                meta={
                    **aggregation_meta(paired_examples),
                    "site": args.site,
                    "target_modality": "image",
                },
            )
            print(
                f"site={args.site} round={current_round} paired_examples={paired_examples} "
                f"train_loss={train_metrics['train_loss']:.4f} empty_update={not bool(params)}",
                flush=True,
            )
            flare.send(output_model)
    finally:
        flare.shutdown()


if __name__ == "__main__":
    main()
