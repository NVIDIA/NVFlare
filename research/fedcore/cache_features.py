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

"""Create local target-present/target-missing Qwen3-VL feature caches."""

import argparse
from pathlib import Path

from src.features import create_feature_cache


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache FedCoRe features with Qwen3-VL.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    return parser


def main() -> None:
    args = define_parser().parse_args()
    from src.qwen_backend import QwenFeatureExtractor

    extractor = QwenFeatureExtractor(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        batch_size=args.batch_size,
        adapter_checkpoint=args.adapter_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    metadata = create_feature_cache(
        data_dir=Path(args.data_dir),
        cache_dir=Path(args.cache_dir),
        qwen_extractor=extractor,
    )
    print(f"Cached Qwen3-VL features with input_dim={metadata['input_dim']} " f"under {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
