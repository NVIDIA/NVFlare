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

import argparse
import json
from pathlib import Path


def define_parser():
    parser = argparse.ArgumentParser(description="Prepare synthetic JSONL data for the Hello HuggingFace example")
    parser.add_argument("--data_root", type=str, default="/tmp/nvflare/hello-huggingface/data")
    parser.add_argument("--n_clients", type=int, default=2)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def prepare_site_data(data_root: Path, client_names: list[str]):
    for idx, site_name in enumerate(client_names, start=1):
        rows = [
            {
                "instruction": "Summarize the site signal in one sentence.",
                "input": f"Site {idx} observed stable local training loss over two batches.",
                "output": f"Site {idx} reports stable local training loss.",
            },
            {
                "instruction": "Rewrite the sentence in a concise technical style.",
                "input": f"Client {site_name} has four synthetic records for this demonstration.",
                "output": f"{site_name} uses four synthetic demonstration records.",
            },
            {
                "instruction": "Classify the deployment mode.",
                "input": "The FL client exchanges model weights through NVFlare and trains locally with Qwen.",
                "output": "This is federated fine-tuning.",
            },
            {
                "instruction": "Extract the relevant framework.",
                "input": "The trainer is patched with nvflare.client.hf before the round loop.",
                "output": "The relevant framework is HuggingFace Trainer.",
            },
        ]
        valid_rows = [
            {
                "instruction": "Summarize the evaluation setup.",
                "input": f"{site_name} evaluates the global Qwen model before local training.",
                "output": f"{site_name} runs pre-train global-model evaluation.",
            }
        ]
        write_jsonl(data_root / site_name / "train.jsonl", rows)
        write_jsonl(data_root / site_name / "valid.jsonl", valid_rows)


def main():
    args = define_parser()
    data_root = Path(args.data_root).expanduser().resolve()
    client_names = [f"site-{idx}" for idx in range(1, args.n_clients + 1)]
    prepare_site_data(data_root, client_names)
    print(f"Prepared synthetic data for {args.n_clients} clients under {data_root}")


if __name__ == "__main__":
    main()
