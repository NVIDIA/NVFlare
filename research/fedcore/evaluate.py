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

"""Select a completion strength on validation data and evaluate once on test data."""

import argparse
import json
import math
from pathlib import Path

import torch
from model import LogitCompletionModel
from src.evaluation import evaluate_sites, prepare_site, select_alpha_from_statistics, validation_sufficient_statistics
from src.features import load_cache_metadata, load_cache_split
from src.validation import non_negative_float, parse_alpha_grid, positive_int, probability


def _load_model(checkpoint: Path, input_dim: int, hidden_dim: int, dropout: float) -> LogitCompletionModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Could not find model parameters in {checkpoint}")
    clean_state = {}
    for key, value in state_dict.items():
        clean = key[6:] if key.startswith("model.") else key
        clean_state[clean] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    model = LogitCompletionModel(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-select and evaluate a FedCoRe completion checkpoint.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=positive_int, default=128)
    parser.add_argument("--dropout", type=probability, default=0.1)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--aggregate-loss-tolerance", type=non_negative_float, default=0.0)
    parser.add_argument("--client-auroc-tolerance", type=non_negative_float, default=0.0)
    return parser


def main() -> None:
    args = define_parser().parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    metadata = load_cache_metadata(cache_dir)
    model = _load_model(
        Path(args.checkpoint).expanduser().resolve(),
        input_dim=int(metadata["input_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = ["site-1", "site-2", "site-3"]

    # Test caches are intentionally not opened until source/scale selection is complete.
    validation_sites = [prepare_site(site, load_cache_split(cache_dir, site, "val"), model) for site in sites]
    site_statistics = validation_sufficient_statistics(validation_sites, alpha_grid)
    selected_alpha, candidate_table = select_alpha_from_statistics(
        site_statistics,
        aggregate_loss_tolerance=args.aggregate_loss_tolerance,
        client_auroc_tolerance=args.client_auroc_tolerance,
    )
    test_sites = [prepare_site(site, load_cache_split(cache_dir, site, "test"), model) for site in sites]
    summary, per_site = evaluate_sites(test_sites, selected_alpha)
    train_sites = metadata["sites"]
    contributing_clients = [site for site in sites if int(train_sites[site]["train"]["paired_examples"]) > 0]
    paired_train_examples = sum(int(train_sites[site]["train"]["paired_examples"]) for site in contributing_clients)
    summary.update(
        {
            "cache_dir": str(cache_dir),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "selection_split": "val",
            "evaluation_split": "test",
            "alpha_grid": alpha_grid,
            "aggregate_loss_tolerance": args.aggregate_loss_tolerance,
            "client_auroc_tolerance": args.client_auroc_tolerance,
            "validation_candidates": candidate_table,
            "validation_site_statistics": site_statistics,
            "contributing_clients": contributing_clients,
            "num_contributing_clients": len(contributing_clients),
            "paired_train_examples": paired_train_examples,
            "feature_metadata": metadata,
        }
    )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "per_site_metrics.json").open("w") as f:
        json.dump(_json_safe(per_site), f, indent=2, sort_keys=True)
        f.write("\n")

    print("FedCoRe MNIST evaluation")
    print("+----------------------+----------+----------+----------+")
    print("| Scope                | Before   | After    | Delta    |")
    print("+----------------------+----------+----------+----------+")
    print(
        f"| Missing-image AUROC  | {summary['missing_before']['auroc']:8.4f} | "
        f"{summary['missing_after']['auroc']:8.4f} | {summary['missing_delta_auroc']:+8.4f} |"
    )
    print(
        f"| Aggregate AUROC      | {summary['aggregate_before']['auroc']:8.4f} | "
        f"{summary['aggregate_after']['auroc']:8.4f} | {summary['aggregate_delta_auroc']:+8.4f} |"
    )
    print("+----------------------+----------+----------+----------+")
    print(f"  selected alpha:       {selected_alpha:.2f}")
    print(f"  contributing clients: {', '.join(contributing_clients)} ({paired_train_examples} paired records)")
    print(f"  summary:                    {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
