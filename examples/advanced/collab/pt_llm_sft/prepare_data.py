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

"""Prepare small per-site instruction-tuning datasets for the Collab LLM example."""

import argparse
import json
from pathlib import Path

DEFAULT_DATA_ROOT = "/tmp/nvflare/collab/pt_llm_sft/data"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row))
            stream.write("\n")


def make_rows(site_name: str, site_number: int) -> tuple[list[dict], list[dict]]:
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


def prepare_data(data_root: Path, num_clients: int) -> None:
    if num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

    for site_number in range(1, num_clients + 1):
        site_name = f"site-{site_number}"
        train_rows, valid_rows = make_rows(site_name, site_number)
        write_jsonl(data_root / site_name / "train.jsonl", train_rows)
        write_jsonl(data_root / site_name / "valid.jsonl", valid_rows)

    manifest = {
        "num_clients": num_clients,
        "sites": [f"site-{site_number}" for site_number in range(1, num_clients + 1)],
    }
    with (data_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-clients", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    prepare_data(data_root, args.num_clients)
    print(f"Prepared data for {args.num_clients} clients under {data_root}")


if __name__ == "__main__":
    main()
