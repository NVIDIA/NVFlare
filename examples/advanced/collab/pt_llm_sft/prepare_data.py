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

"""Prepare per-site instruction-tuning datasets for the Collab LLM example."""

import argparse
import json
from pathlib import Path

DEFAULT_DATA_ROOT = "/tmp/nvflare/collab/pt_llm_sft/data"
DOLLY_DATASET_NAME = "databricks/databricks-dolly-15k"
DOLLY_DATASET_SPLIT = "train"
DATA_MODES = ("synthetic", "dolly")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row))
            stream.write("\n")


def write_manifest(data_root: Path, manifest: dict) -> None:
    with (data_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def site_names(num_clients: int) -> list[str]:
    return [f"site-{site_number}" for site_number in range(1, num_clients + 1)]


def make_synthetic_rows(site_name: str, site_number: int) -> tuple[list[dict], list[dict]]:
    train_rows = [
        {
            "instruction": "Summarize the client update in one sentence.",
            "input": f"{site_name} completed local instruction tuning on its private records.",
            "output": f"{site_name} completed its local instruction-tuning update.",
        },
        {
            "instruction": "Classify the learning setup.",
            "input": "Several sites train locally and a coordinator averages their model parameters.",
            "output": "This is federated learning with full-model averaging.",
        },
        {
            "instruction": "Rewrite this in a concise technical style.",
            "input": f"Client {site_number} returns its full language-model state after local training.",
            "output": f"Client {site_number} returns the locally trained full-model state.",
        },
        {
            "instruction": "State the benefit of direct object exchange.",
            "input": "The training function returns a dictionary of PyTorch tensors to the server.",
            "output": "Application code does not manually serialize or deserialize model parameters.",
        },
        {
            "instruction": "Identify what is aggregated.",
            "input": "Every model parameter is trainable during the local supervised fine-tuning interval.",
            "output": "The server aggregates the complete trained model state.",
        },
        {
            "instruction": "Explain simulation in one sentence.",
            "input": "The server and clients run as a federated workflow on one local machine.",
            "output": "Simulation exercises the federated workflow locally with multiple logical sites.",
        },
    ]
    valid_rows = [
        {
            "instruction": "Summarize the global-model evaluation.",
            "input": f"{site_name} evaluates the received global model before local training.",
            "output": f"{site_name} measures the global model before updating it.",
        }
    ]
    return train_rows, valid_rows


def normalize_dolly_row(row: dict) -> dict:
    return {
        "instruction": row["instruction"],
        "input": row.get("context") or "",
        "output": row["response"],
    }


def prepare_synthetic_data(data_root: Path, num_clients: int) -> dict:
    counts = {}
    for site_number, site_name in enumerate(site_names(num_clients), start=1):
        train_rows, valid_rows = make_synthetic_rows(site_name, site_number)
        write_jsonl(data_root / site_name / "train.jsonl", train_rows)
        write_jsonl(data_root / site_name / "valid.jsonl", valid_rows)
        counts[site_name] = {"train": len(train_rows), "valid": len(valid_rows)}

    return {
        "data_mode": "synthetic",
        "num_clients": num_clients,
        "sites": site_names(num_clients),
        "site_counts": counts,
    }


def prepare_dolly_data(
    data_root: Path,
    num_clients: int,
    validation_fraction: float,
    seed: int,
    cache_dir: Path | None,
) -> dict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    import datasets

    load_kwargs = {
        "path": DOLLY_DATASET_NAME,
        "split": DOLLY_DATASET_SPLIT,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    dataset = datasets.load_dataset(**load_kwargs)
    if len(dataset) < num_clients * 2:
        raise ValueError(f"{DOLLY_DATASET_NAME} has {len(dataset)} rows; need at least two rows per client")

    source_fingerprint = getattr(dataset, "_fingerprint", None)
    dataset_version = str(dataset.info.version) if dataset.info.version is not None else None
    dataset = dataset.shuffle(seed=seed)
    counts = {}

    for site_index, site_name in enumerate(site_names(num_clients)):
        site_dataset = dataset.shard(num_shards=num_clients, index=site_index, contiguous=True)
        valid_count = max(1, int(len(site_dataset) * validation_fraction))
        valid_count = min(valid_count, len(site_dataset) - 1)
        train_count = len(site_dataset) - valid_count
        train_rows = [normalize_dolly_row(row) for row in site_dataset.select(range(train_count))]
        valid_rows = [normalize_dolly_row(row) for row in site_dataset.select(range(train_count, len(site_dataset)))]
        write_jsonl(data_root / site_name / "train.jsonl", train_rows)
        write_jsonl(data_root / site_name / "valid.jsonl", valid_rows)
        counts[site_name] = {"train": len(train_rows), "valid": len(valid_rows)}

    return {
        "data_mode": "dolly",
        "dataset_name": DOLLY_DATASET_NAME,
        "dataset_split": DOLLY_DATASET_SPLIT,
        "dataset_version": dataset_version,
        "source_fingerprint": source_fingerprint,
        "selected_fingerprint": getattr(dataset, "_fingerprint", None),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "num_clients": num_clients,
        "sites": site_names(num_clients),
        "site_counts": counts,
    }


def prepare_data(
    data_root: Path,
    num_clients: int,
    data_mode: str = "synthetic",
    validation_fraction: float = 0.1,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> dict:
    if num_clients < 1:
        raise ValueError("--num-clients must be at least 1")
    if data_mode == "synthetic":
        manifest = prepare_synthetic_data(data_root, num_clients)
    elif data_mode == "dolly":
        manifest = prepare_dolly_data(data_root, num_clients, validation_fraction, seed, cache_dir)
    else:
        raise ValueError(f"unsupported data mode: {data_mode}")
    write_manifest(data_root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--data-mode", choices=DATA_MODES, default="synthetic")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", help="Optional Hugging Face dataset cache directory for Dolly")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    manifest = prepare_data(
        data_root,
        args.num_clients,
        data_mode=args.data_mode,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    print(f"Prepared {manifest['data_mode']} data for {args.num_clients} clients under {data_root}")


if __name__ == "__main__":
    main()
