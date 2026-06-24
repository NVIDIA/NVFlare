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
"""Create a tiny deterministic instruction-following split for SFT smoke tests."""

from __future__ import annotations

import argparse
import json
import os

SITE_EXAMPLES = {
    "site-1": [
        (
            "Explain federated learning in one sentence.",
            "Federated learning trains models across sites without moving raw data.",
        ),
        (
            "Name one benefit of privacy-preserving training.",
            "It helps organizations collaborate while keeping sensitive data local.",
        ),
        (
            "Summarize what an aggregation round does.",
            "An aggregation round combines client model updates into a new global model.",
        ),
        (
            "Define a local training step.",
            "A local training step updates a client model using that client's own examples.",
        ),
    ],
    "site-2": [
        (
            "Write a short answer about model checkpoints.",
            "A model checkpoint stores weights that can be loaded later.",
        ),
        (
            "What does a client send to a federated server?",
            "A client sends a model update or model weights, not its raw data.",
        ),
        (
            "Why run clients sequentially in a demo?",
            "Sequential clients reduce peak GPU memory during local simulation.",
        ),
        ("Describe full-model SFT briefly.", "Full-model SFT updates all trainable model weights on instruction data."),
    ],
    "site-3": [
        (
            "Give one sentence about validation data.",
            "Validation data estimates model quality without updating the weights.",
        ),
        ("What is the role of FedAvg?", "FedAvg averages client contributions to produce the next global model."),
        (
            "State why deterministic data helps examples.",
            "Deterministic data makes runs easier to reproduce and compare.",
        ),
        ("What should an SFT response be?", "An SFT response should directly answer the instruction."),
    ],
}

VALIDATION_EXAMPLES = [
    ("What is federated learning?", "Federated learning trains a shared model while data remains at each site."),
    ("What does the server aggregate?", "The server aggregates model updates from participating clients."),
    ("Why use a validation file?", "A validation file checks model quality during training."),
]

TEST_EXAMPLES = [
    ("Explain FedAvg in one sentence.", "FedAvg averages client model updates into a global model."),
    ("What stays local in federated learning?", "The raw training data stays local to each client."),
]


def _write_jsonl(path: str, examples: list[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for instruction, response in examples:
            f.write(json.dumps({"input": instruction, "output": response}) + "\n")


def define_parser():
    parser = argparse.ArgumentParser(description="Create deterministic synthetic SFT JSONL files.")
    parser.add_argument("--out_dir", default="data/synthetic_sft")
    return parser.parse_args()


def main():
    args = define_parser()
    for site_name, examples in SITE_EXAMPLES.items():
        _write_jsonl(os.path.join(args.out_dir, f"{site_name}_train.jsonl"), examples)
    _write_jsonl(os.path.join(args.out_dir, "validation.jsonl"), VALIDATION_EXAMPLES)
    _write_jsonl(os.path.join(args.out_dir, "test.jsonl"), TEST_EXAMPLES)
    print(f"Wrote synthetic SFT data to {args.out_dir}")


if __name__ == "__main__":
    main()
