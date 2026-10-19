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

"""Cache the pretrained model and prepare two disjoint WikiText-2 training shards."""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def pack_text(split, tokenizer, seq_length):
    """Pack nonempty rows into fixed-length blocks, separating rows with EOS."""
    texts = [text for text in split["text"] if text.strip()]
    if not texts:
        raise ValueError("The text split contains no nonempty rows.")
    tokens = []
    for row in tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]:
        tokens.extend(row)
        tokens.append(tokenizer.eos_token_id)
    # Drop only the final incomplete block; no padding or padded labels are needed.
    stop = len(tokens) - len(tokens) % seq_length
    blocks = [tokens[start : start + seq_length] for start in range(0, stop, seq_length)]
    if len(blocks) < 2:
        raise ValueError(f"Need at least two complete {seq_length}-token blocks; found {len(blocks)}.")
    return Dataset.from_dict({"input_ids": blocks})


def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 for two FSDP2 clients.")
    parser.add_argument("--data_dir", default="/tmp/nvflare/fsdp2/data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seq_length", type=int, default=256)
    args = parser.parse_args()
    if args.seq_length < 2:
        parser.error("seq_length must be at least 2 for causal language modeling")

    # Cache checkpoint files without allocating a model in CPU or GPU memory.
    if not Path(args.model_name).is_dir():
        snapshot_download(
            args.model_name,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges.txt"],
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token to separate text rows.")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    splits = {
        "site-1/train": dataset["train"].shard(num_shards=2, index=0, contiguous=True),
        "site-2/train": dataset["train"].shard(num_shards=2, index=1, contiguous=True),
        "validation": dataset["validation"],
    }
    for name, split in splits.items():
        packed = pack_text(split, tokenizer, args.seq_length)
        path = Path(args.data_dir) / name
        packed.save_to_disk(str(path))
        print(f"{name}: {len(packed):,} blocks of {args.seq_length} tokens -> {path}")


if __name__ == "__main__":
    main()
