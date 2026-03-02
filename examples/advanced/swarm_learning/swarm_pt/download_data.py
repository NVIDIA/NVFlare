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

"""Download and cache the wikitext-2-raw-v1 dataset and the Qwen2.5-0.5B model.

Run once before starting the simulation:

    python download_data.py
    python download_data.py --skip_model        # dataset only
    python download_data.py --model_path /my/local/path   # skip HF model download

Both downloads are licence-free:
  - wikitext-2-raw-v1  (Apache-2.0)
  - Qwen/Qwen2.5-0.5B  (Apache-2.0, no HuggingFace gating)
"""

import argparse


def download_dataset(cache_dir=None):
    from datasets import load_dataset

    print("Downloading wikitext-2-raw-v1 (train / validation / test splits) ...")
    for split in ("train", "validation", "test"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
        print(f"  {split}: {len(ds):,} examples")
    print("Dataset download complete.\n")


def download_model(model_path: str, cache_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading model '{model_path}' ...")
    AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    print("Model download complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-download dataset and model for the swarm LoRA example.")
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace Hub model ID or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--skip_model",
        action="store_true",
        help="Skip model download (useful when the model is already cached or on a local mount)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional HuggingFace cache directory (defaults to ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    download_dataset(cache_dir=args.cache_dir)

    if not args.skip_model:
        download_model(args.model_path, cache_dir=args.cache_dir)
    else:
        print(f"Skipping model download (--skip_model). Using '{args.model_path}' as-is.")

    print("All downloads complete. You can now run: python job.py")


if __name__ == "__main__":
    main()
