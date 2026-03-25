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
Run MedGemma inference on prepared histopathology JSON records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from data_utils import DEFAULT_MODEL_NAME_OR_PATH, TISSUE_CLASSES, resolve_image_path
from inference_utils import load_model_and_processor, predict_label
from PIL import Image


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _load_records(data_file: str, image_root: str, max_samples: int):
    with open(data_file, "r") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        records = [records]

    samples = []
    for record in records[:max_samples]:
        image_path = resolve_image_path(record["image"], image_root)
        if not os.path.isfile(image_path):
            print(f"Warning: image not found: {image_path}", file=sys.stderr)
            continue
        sample = dict(record)
        sample["image"] = Image.open(image_path).convert("RGB")
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma inference on prepared site JSON records.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Model ID, adapter checkpoint directory, or NVFlare FL_global_model.pt path.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/site-1/validation.json",
        help="Prepared JSON file from prepare_data.py (default: ./data/site-1/validation.json).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="./NCT-CRC-HE-100K",
        help="Root directory used to resolve relative image paths in data_file.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Base MedGemma model used when model_path points to an adapter directory or FL_global_model.pt.",
    )
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to run (default: 5).")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum new tokens per response (default: 40).",
    )
    parser.add_argument("--device", type=str, default="cuda", help='Target device, e.g. "cuda" or "cpu".')
    args = parser.parse_args()

    data_file = _abs_path(args.data_file)
    image_root = _abs_path(args.image_root)
    model_path = args.model_path if not os.path.exists(args.model_path) else _abs_path(args.model_path)

    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    samples = _load_records(data_file, image_root, args.max_samples)
    if not samples:
        print("No valid samples found. Check --data_file and --image_root.")
        return 1

    model, processor = load_model_and_processor(model_path, args.base_model, args.device)
    print(f"Running inference on {len(samples)} sample(s)\n")

    for index, sample in enumerate(samples, start=1):
        response_text, predicted_index, predicted_label = predict_label(
            model=model,
            processor=processor,
            image=sample["image"],
            max_new_tokens=args.max_new_tokens,
        )
        ground_truth = sample.get("label_name") or TISSUE_CLASSES[int(sample["label"])]

        print(f"--- Sample {index} ---")
        print(f"Ground truth: {ground_truth}")
        print(f"Prediction:   {predicted_label}")
        print(f"Raw output:   {response_text[:240]}{'...' if len(response_text) > 240 else ''}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
