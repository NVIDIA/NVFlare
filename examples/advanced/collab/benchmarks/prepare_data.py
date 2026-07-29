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

"""Prepare the datasets used by the Collab benchmark configurations."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from collab.pt_llm_sft.prepare_data import prepare_data as prepare_sft_data
from collab.pt_llm_sft.prepare_data import write_jsonl
from collab.pt_llm_sft.pt_llm_sft import validate_prepared_data
from torchvision import datasets as tv_datasets

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_dolly_row(row: dict) -> dict:
    return {
        "instruction": row["instruction"],
        "input": row.get("context", ""),
        "output": row["response"],
    }


def prepare_dolly_data(data_root: Path, config: dict) -> None:
    num_clients = config["num_clients"]
    train_per_client = config["train_examples_per_client"]
    valid_per_client = config["valid_examples_per_client"]
    total_per_client = train_per_client + valid_per_client
    total_required = num_clients * total_per_client
    if min(train_per_client, valid_per_client) < 1:
        raise ValueError("Dolly train and validation counts must be positive")
    if train_per_client < config["syncs_per_epoch"]:
        raise ValueError("Dolly training examples per client must be at least syncs_per_epoch")

    if config.get("hf_home"):
        os.environ["HF_HOME"] = config["hf_home"]
    import datasets as hf_datasets

    dataset = hf_datasets.load_dataset(config["dataset_name"], split=config["dataset_split"])
    if len(dataset) < total_required:
        raise ValueError(f"{config['dataset_name']} has {len(dataset)} rows; need {total_required}")
    source_fingerprint = getattr(dataset, "_fingerprint", None)
    dataset_version = str(dataset.info.version) if dataset.info.version is not None else None
    dataset = dataset.shuffle(seed=config["data_seed"]).select(range(total_required))

    hashes = {}
    sites = []
    for site_index in range(num_clients):
        site_name = f"site-{site_index + 1}"
        sites.append(site_name)
        offset = site_index * total_per_client
        site_rows = dataset.select(range(offset, offset + total_per_client))
        train_rows = [normalize_dolly_row(site_rows[index]) for index in range(train_per_client)]
        valid_rows = [normalize_dolly_row(site_rows[index]) for index in range(train_per_client, total_per_client)]
        for filename, rows in (("train.jsonl", train_rows), ("valid.jsonl", valid_rows)):
            path = data_root / site_name / filename
            write_jsonl(path, rows)
            hashes[str(path.relative_to(data_root))] = file_sha256(path)

    manifest = {
        "dataset_name": config["dataset_name"],
        "dataset_split": config["dataset_split"],
        "dataset_version": dataset_version,
        "source_fingerprint": source_fingerprint,
        "selected_fingerprint": getattr(dataset, "_fingerprint", None),
        "data_seed": config["data_seed"],
        "num_clients": num_clients,
        "train_examples_per_client": train_per_client,
        "valid_examples_per_client": valid_per_client,
        "sites": sites,
        "sha256": hashes,
    }
    with (data_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_data_identity(data_root: Path, config: dict) -> None:
    validate_prepared_data(data_root, config["num_clients"])
    if not config.get("dataset_name"):
        return
    with (data_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = {
        "dataset_name": config["dataset_name"],
        "dataset_split": config["dataset_split"],
        "data_seed": config["data_seed"],
        "train_examples_per_client": config["train_examples_per_client"],
        "valid_examples_per_client": config["valid_examples_per_client"],
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ValueError(f"prepared data manifest differs for: {', '.join(mismatches)}")
    for relative_path, expected_hash in manifest.get("sha256", {}).items():
        path = data_root / relative_path
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"prepared data hash differs for: {relative_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("all", "pt_llm_sft", "simple_split_learning"),
        default="all",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "pt_llm_sft.json",
        help="SFT benchmark config; use pt_llm_sft_slurm.json on the cluster",
    )
    args = parser.parse_args()

    if args.workload in ("all", "pt_llm_sft"):
        config = load_config(args.config.expanduser().resolve())
        data_root = Path(config["data_root"])
        data_root.mkdir(parents=True, exist_ok=True)
        try:
            validate_data_identity(data_root, config)
            print(f"Reusing prepared SFT data under {data_root}")
        except (FileNotFoundError, ValueError) as error:
            if any(data_root.iterdir()):
                raise RuntimeError(
                    f"refusing to overwrite incompatible data under {data_root}; " "select an empty benchmark data root"
                ) from error
            if config.get("dataset_name"):
                prepare_dolly_data(data_root, config)
            else:
                prepare_sft_data(data_root, config["num_clients"])
            print(f"Prepared SFT data for {config['num_clients']} clients under {data_root}")

    if args.workload in ("all", "simple_split_learning"):
        config = load_config(CONFIG_DIR / "simple_split_learning.json")
        data_root = Path(config["data_root"])
        print(f"Downloading/verifying MNIST under {data_root} ...")
        tv_datasets.MNIST(root=str(data_root), train=True, download=True)
        print(f"Prepared MNIST under {data_root}")


if __name__ == "__main__":
    main()
