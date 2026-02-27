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
Prepare PubMedVision for federated Qwen3-VL by splitting into 3 parts (one per client).

Supports:
  - Local JSON/JSONL file (e.g. from git clone of the dataset repo)
  - HuggingFace Hub (load_dataset)

Writes train.json as a single JSON array so Qwen3-VL train_qwen.py (which uses
json.load for .json) can read it. HuggingFace to_json() writes JSONL by default.

Also injects <image> placeholders into the first user message so Qwen's
_build_messages() consumes each image (one placeholder per image required).
"""

import argparse
import json
import os


def _ensure_qwen_image_placeholders(record):
    """Inject <image> placeholders into the first human turn per Qwen format.

    Single image:  value = "<image>" + newline + question
    Multi-image:   value = "<image>" + newline + "<image>" + newline + question
    (one <image> per image). See Qwen3-VL dataset config single/multi-image examples.
    """
    image_key = "image"
    images = record.get(image_key) or []
    if isinstance(images, str):
        images = [images]
    if not images:
        return record
    # One "<image>\n" per image, then the original question (Qwen single/multi-image format)
    placeholders = "".join("<image>\n" for _ in images)
    convs = list(record.get("conversations", []))
    new_conversations = []
    first_human_done = False
    for turn in convs:
        if turn.get("from") == "human" and not first_human_done:
            first_human_done = True
            value = turn.get("value", "")
            # Only inject placeholders if the prompt does not already contain <image>
            if "<image>" not in value:
                turn = {**turn, "value": placeholders + value}
        new_conversations.append(turn)
    return {**record, "conversations": new_conversations}


def main():
    parser = argparse.ArgumentParser(description="Prepare PubMedVision for federated Qwen3-VL (3 clients)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Root directory for client data (default: ./data)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of client splits (default: 3)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Max samples per client (default: None = use full split). Set e.g. 5000 for a quick run.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="PubMedVision/PubMedVision_InstructionTuning_VQA.json",
        help="Path to local PubMedVision JSON (default: PubMedVision/PubMedVision_InstructionTuning_VQA.json, "
        "after download_data.py). Set to empty string to load from HuggingFace Hub instead.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="PubMedVision_InstructionTuning_VQA",
        help="PubMedVision subset when loading from Hub: PubMedVision_Alignment_VQA or "
        "PubMedVision_InstructionTuning_VQA (ignored if --data_file is set)",
    )
    args = parser.parse_args()
    if args.num_clients <= 0:
        raise ValueError(f"--num_clients must be > 0, got {args.num_clients}")
    if args.subset_size is not None and args.subset_size <= 0:
        raise ValueError(f"--subset_size must be > 0 when set, got {args.subset_size}")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    if args.data_file and args.data_file.strip():
        data_file = args.data_file.strip()
        if not os.path.isfile(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}. Run download_data.py first, or use --data_file '' to load from HuggingFace Hub."
            )
        print(f"Loading from local file: {data_file}")
        dataset = load_dataset("json", data_files=data_file, split="train")
    else:
        print(f"Loading {args.split_name} from HuggingFace ...")
        dataset = load_dataset(
            "FreedomIntelligence/PubMedVision",
            args.split_name,
            split="train",
            trust_remote_code=True,
        )

    n_total = len(dataset)
    n_clients = args.num_clients
    n_per_client_full = n_total // n_clients
    if n_per_client_full == 0:
        raise ValueError(
            f"Insufficient data: dataset has {n_total} samples but {n_clients} clients requested. "
            f"Each client would receive 0 samples. Use fewer clients or more data."
        )
    if args.subset_size is not None:
        n_per_client = min(n_per_client_full, args.subset_size)
        if n_per_client <= 0:
            raise ValueError(
                f"Insufficient data: subset_size={args.subset_size} results in 0 samples per client. "
                f"Use a larger subset_size (max {n_per_client_full} per client)."
            )
    else:
        n_per_client = n_per_client_full

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(n_clients):
        start = i * (n_total // n_clients)
        end = start + n_per_client
        shard = dataset.select(range(start, end))
        client_dir = os.path.join(args.output_dir, f"site-{i + 1}")
        os.makedirs(client_dir, exist_ok=True)
        out_path = os.path.join(client_dir, "train.json")
        # Single JSON array (not JSONL) so Qwen train_qwen.py json.load() works
        records = [shard[i] for i in range(len(shard))]
        # Inject <image> placeholders so Qwen _build_messages consumes each image
        records = [_ensure_qwen_image_placeholders(r) for r in records]
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  site-{i + 1}: {len(shard)} samples -> {out_path}")

    print(f"Done. Client data under {args.output_dir}/site-1, site-2, site-3.")


if __name__ == "__main__":
    main()
