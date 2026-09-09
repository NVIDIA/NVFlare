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

"""Run the matched CIFAR-10 comparison requested for the research draft."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from protocol import PROTOCOL_VERSION, canonical_config_hash, common_run_config, method_run_config

DEFAULT_METHODS = ("fedavg", "fedprox", "scaffold", "fedce", "adaptive")
DEFAULT_SEEDS = (7, 19, 31, 43, 57)
DEFAULT_ALPHAS = (0.1, 0.5)
DEFAULT_PARTICIPATION = (1.0, 0.75)


def _expected_common_hash(args) -> str:
    return canonical_config_hash(
        common_run_config(
            n_clients=args.n_clients,
            num_rounds=args.num_rounds,
            aggregation_epochs=args.aggregation_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            validation_fraction=args.validation_fraction,
        )
    )


def _expected_method_hash(args, method: str) -> str:
    return canonical_config_hash(
        method_run_config(
            method,
            fedprox_mu=args.fedprox_mu,
            fedce_mode=args.fedce_mode,
            sample_exponent=args.sample_exponent,
            representation_exponent=args.representation_exponent,
            metric_prior_strength=args.metric_prior_strength,
            max_blend_factor=args.max_blend_factor,
            activation_warmup_rounds=args.activation_warmup_rounds,
            activation_patience=args.activation_patience,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            allow_changing_cohort_evidence=args.allow_changing_cohort_evidence,
        )
    )


def _completed_keys(path: Path, args) -> set[tuple]:
    """Return completed rows only when their full configuration matches this campaign."""

    if not path.is_file():
        return set()
    common_hash = _expected_common_hash(args)
    method_hashes = {method: _expected_method_hash(args, method) for method in args.methods}
    keys = set()
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method", ""))
            if row.get("protocol_version") != PROTOCOL_VERSION or method not in method_hashes:
                continue
            if row.get("common_config_hash") != common_hash:
                continue
            if row.get("method_config_hash") != method_hashes[method]:
                continue
            keys.add(
                (
                    method,
                    float(row["alpha"]),
                    float(row["participation_rate"]),
                    int(row["seed"]),
                )
            )
    return keys


def _run_command(args, method: str, alpha: float, participation: float, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run.py"),
        "--method",
        method,
        "--alpha",
        str(alpha),
        "--participation_rate",
        str(participation),
        "--seed",
        str(seed),
        "--validation_fraction",
        str(args.validation_fraction),
        "--n_clients",
        str(args.n_clients),
        "--num_rounds",
        str(args.num_rounds),
        "--aggregation_epochs",
        str(args.aggregation_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num_workers",
        str(args.num_workers),
        "--workspace_root",
        args.workspace_root,
        "--split_root",
        args.split_root,
        "--results_jsonl",
        args.results_jsonl,
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--eval_num_workers",
        str(args.eval_num_workers),
    ]
    if args.num_threads is not None:
        command.extend(["--num_threads", str(args.num_threads)])
    if args.gpu_config:
        command.extend(["--gpu_config", args.gpu_config])
    if args.eval_device:
        command.extend(["--eval_device", args.eval_device])
    if method == "fedprox":
        command.extend(["--fedprox_mu", str(args.fedprox_mu)])
    if method == "fedce":
        command.extend(["--fedce_mode", args.fedce_mode])
    if method == "adaptive":
        command.extend(
            [
                "--sample_exponent",
                str(args.sample_exponent),
                "--representation_exponent",
                str(args.representation_exponent),
                "--metric_prior_strength",
                str(args.metric_prior_strength),
                "--max_blend_factor",
                str(args.max_blend_factor),
                "--activation_warmup_rounds",
                str(args.activation_warmup_rounds),
                "--activation_patience",
                str(args.activation_patience),
                "--min_weight",
                str(args.min_weight),
                "--max_weight",
                str(args.max_weight),
            ]
        )
        if args.allow_changing_cohort_evidence:
            command.append("--allow_changing_cohort_evidence")
    return command


def main(args):
    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and results_path.exists():
        results_path.unlink()

    completed = _completed_keys(results_path, args) if args.resume else set()
    planned = []
    for alpha in args.alphas:
        for participation in args.participation_rates:
            for method in args.methods:
                if method == "fedopt" and participation < 1.0:
                    continue
                for seed in args.seeds:
                    key = (method, float(alpha), float(participation), int(seed))
                    if key in completed:
                        continue
                    planned.append((method, float(alpha), float(participation), int(seed)))

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Common configuration hash: {_expected_common_hash(args)}")
    print(f"Planned CIFAR-10 runs: {len(planned)}")
    for index, (method, alpha, participation, seed) in enumerate(planned, start=1):
        command = _run_command(args, method, alpha, participation, seed)
        print(
            f"[{index}/{len(planned)}] method={method} alpha={alpha} "
            f"participation={participation} seed={seed}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)

    if args.dry_run:
        return

    summary_command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "summarize_results.py"),
        "--input",
        args.results_jsonl,
        "--output",
        args.summary_json,
        "--markdown_output",
        args.results_markdown,
        "--reference_method",
        "adaptive",
        "--protocol_version",
        PROTOCOL_VERSION,
        "--common_config_hash",
        _expected_common_hash(args),
        "--require_complete",
        "--methods",
        *args.methods,
        "--alphas",
        *[str(value) for value in args.alphas],
        "--participation_rates",
        *[str(value) for value in args.participation_rates],
        "--seeds",
        *[str(value) for value in args.seeds],
    ]
    for method in args.methods:
        summary_command.extend(["--method_config_hash", f"{method}={_expected_method_hash(args, method)}"])
    subprocess.run(summary_command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--participation_rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_PARTICIPATION),
    )
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--gpu_config", default=None)
    parser.add_argument("--eval_device", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--fedce_mode", choices=("plus", "times"), default="plus")
    parser.add_argument("--workspace_root", default="/tmp/nvflare/adaptive_hetero_cifar10")
    parser.add_argument("--split_root", default="/tmp/cifar10_splits/adaptive_hetero_eval")
    parser.add_argument("--results_jsonl", default="results/cifar10_runs.jsonl")
    parser.add_argument("--summary_json", default="results/cifar10_summary.json")
    parser.add_argument("--results_markdown", default="results/cifar10_main_results.md")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--sample_exponent", type=float, default=0.65)
    parser.add_argument("--representation_exponent", type=float, default=0.70)
    parser.add_argument("--metric_prior_strength", type=float, default=100.0)
    parser.add_argument("--max_blend_factor", type=float, default=0.20)
    parser.add_argument("--activation_warmup_rounds", type=int, default=3)
    parser.add_argument("--activation_patience", type=int, default=2)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--allow_changing_cohort_evidence", action="store_true")
    main(parser.parse_args())
