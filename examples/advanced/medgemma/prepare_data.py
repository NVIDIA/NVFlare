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
"""
Prepare site-based MedGemma training and validation splits from NCT-CRC-HE-100K.

Task setup adapted from the Google Health MedGemma Hugging Face fine-tuning notebook:
https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
"""

from __future__ import annotations

import argparse
import json
import os

from data_utils import (
    collect_image_records,
    split_records_for_clients,
    summarize_label_distribution,
    write_label_distribution_svg,
)


def _format_site_summary(split_data: dict) -> list[str]:
    summary_lines = []
    dominant_labels = split_data.get("dominant_labels")
    if dominant_labels:
        summary_lines.append(f"    dominant_labels: {', '.join(dominant_labels)}")
    summary_lines.append(f"    train_labels: {', '.join(summarize_label_distribution(split_data['train']))}")
    summary_lines.append(f"    val_labels:   {', '.join(summarize_label_distribution(split_data['validation']))}")
    return summary_lines


def main():
    parser = argparse.ArgumentParser(description="Prepare NCT-CRC-HE-100K shards for federated MedGemma fine-tuning.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./NCT-CRC-HE-100K",
        help="Path to the extracted NCT-CRC-HE-100K directory (default: ./NCT-CRC-HE-100K).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Root output directory for site-1, site-2, site-3 splits (default: ./data).",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of client shards to create (default: 3).",
    )
    parser.add_argument(
        "--samples_per_client",
        type=int,
        default=3333,
        help="Total samples per client before the train/validation split. Use 0 to use the full dataset evenly. "
        "Default: 3333, which roughly matches the official 10k-sample notebook subset across 3 clients.",
    )
    parser.add_argument(
        "--validation_size_per_client",
        type=int,
        default=333,
        help="Validation samples reserved from each client shard (default: 333).",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=("heterogeneous", "random"),
        default="heterogeneous",
        help=(
            "Site split strategy. 'heterogeneous' creates class-skewed, non-IID client shards by default; "
            "'random' keeps the older IID-style random sharding."
        ),
    )
    parser.add_argument(
        "--dominant_fraction",
        type=float,
        default=0.8,
        help=(
            "For --split_strategy heterogeneous, fraction of each site shard biased toward its dominant label group "
            "(default: 0.8)."
        ),
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help=(
            "Optional output path for a generated SVG chart of per-client training-label distributions. "
            "Defaults to <output_dir>/split_label_distribution.svg."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    args = parser.parse_args()

    samples_per_client = args.samples_per_client if args.samples_per_client > 0 else None

    records = collect_image_records(args.dataset_dir)
    site_splits = split_records_for_clients(
        records=records,
        num_clients=args.num_clients,
        samples_per_client=samples_per_client,
        validation_size_per_client=args.validation_size_per_client,
        seed=args.seed,
        split_strategy=args.split_strategy,
        dominant_fraction=args.dominant_fraction,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(records)} labeled images under {os.path.abspath(args.dataset_dir)}")
    print(f"Using split_strategy={args.split_strategy}, dominant_fraction={args.dominant_fraction}")
    for site_name, split_data in site_splits.items():
        site_dir = os.path.join(args.output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)

        train_path = os.path.join(site_dir, "train.json")
        validation_path = os.path.join(site_dir, "validation.json")

        with open(train_path, "w") as train_file:
            json.dump(split_data["train"], train_file, indent=2)
        with open(validation_path, "w") as validation_file:
            json.dump(split_data["validation"], validation_file, indent=2)

        print(
            f"  {site_name}: train={len(split_data['train'])} -> {train_path}, "
            f"validation={len(split_data['validation'])} -> {validation_path}"
        )
        for line in _format_site_summary(split_data):
            print(line)

    plot_path = args.plot_path or os.path.join(args.output_dir, "split_label_distribution.svg")
    write_label_distribution_svg(site_splits, plot_path)
    print(f"Wrote label distribution plot to {os.path.abspath(plot_path)}")
    print(f"Done. Client data written under {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
