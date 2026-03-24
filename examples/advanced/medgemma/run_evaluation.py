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
Evaluate MedGemma before and after fine-tuning on CRC-VAL-HE-7K.
"""

from __future__ import annotations

import argparse
import os
import sys

import evaluate
from data_utils import DEFAULT_EVAL_DATASET_DIR, DEFAULT_MODEL_NAME_OR_PATH, collect_image_records, sample_records
from inference_utils import load_model_and_processor, predict_label
from PIL import Image


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _load_eval_records(dataset_dir: str, max_samples: int | None, seed: int):
    records = collect_image_records(dataset_dir)
    return sample_records(records, max_samples=max_samples, seed=seed)


def _evaluate_model(
    label: str,
    model_path: str,
    base_model: str,
    dataset_dir: str,
    device: str,
    records: list[dict],
    max_new_tokens: int,
    show_examples: int,
) -> dict:
    print(f"\nEvaluating {label}: {model_path}")
    model, processor = load_model_and_processor(model_path=model_path, base_model=base_model, device=device)

    predictions = []
    references = []
    unparsed = 0

    for idx, record in enumerate(records, start=1):
        with Image.open(os.path.join(dataset_dir, record["image"])) as image_file:
            image = image_file.convert("RGB")
        response_text, predicted_index, predicted_label = predict_label(
            model=model,
            processor=processor,
            image=image,
            max_new_tokens=max_new_tokens,
        )
        predictions.append(predicted_index)
        references.append(record["label"])
        if predicted_index < 0:
            unparsed += 1

        if idx <= show_examples:
            print(f"--- {label} sample {idx} ---")
            print(f"Ground truth: {record['label_name']}")
            print(f"Prediction:   {predicted_label}")
            print(f"Raw output:   {response_text[:240]}{'...' if len(response_text) > 240 else ''}")

    accuracy_metric = evaluate.load("accuracy")
    metrics = accuracy_metric.compute(predictions=predictions, references=references)
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    metrics["correct"] = correct
    metrics["total"] = len(references)
    metrics["unparsed"] = unparsed
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma accuracy before and after fine-tuning.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=f"./{DEFAULT_EVAL_DATASET_DIR}",
        help="Path to the extracted CRC-VAL-HE-7K directory (default: ./CRC-VAL-HE-7K).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Base model path or ID used for the before-fine-tuning evaluation.",
    )
    parser.add_argument(
        "--tuned_model_path",
        type=str,
        required=True,
        help="Fine-tuned adapter directory or NVFlare FL_global_model.pt path used for the after-fine-tuning evaluation.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Base MedGemma model ID used when loading adapter-only tuned checkpoints.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum evaluation samples. The notebook uses 1000 by default (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for evaluation subset selection.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum generated tokens per sample (default: 40).",
    )
    parser.add_argument("--device", type=str, default="cuda", help='Target device, e.g. "cuda" or "cpu".')
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Number of qualitative examples to print per model before the summary metrics (default: 0).",
    )
    args = parser.parse_args()

    dataset_dir = _abs_path(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_dir}. "
            "Download CRC-VAL-HE-7K first, for example via `python download_data.py --include_eval`."
        )

    records = _load_eval_records(dataset_dir=dataset_dir, max_samples=args.max_samples, seed=args.seed)
    if not records:
        print("No evaluation samples found.")
        return 1
    print(f"Using {len(records)} evaluation sample(s) from {dataset_dir}")

    base_metrics = _evaluate_model(
        label="Base model",
        model_path=args.base_model_path,
        base_model=args.base_model,
        dataset_dir=dataset_dir,
        device=args.device,
        records=records,
        max_new_tokens=args.max_new_tokens,
        show_examples=args.show_examples,
    )
    tuned_metrics = _evaluate_model(
        label="Fine-tuned model",
        model_path=args.tuned_model_path,
        base_model=args.base_model,
        dataset_dir=dataset_dir,
        device=args.device,
        records=records,
        max_new_tokens=args.max_new_tokens,
        show_examples=args.show_examples,
    )

    print("\nAccuracy summary")
    print(
        f"Base model:       accuracy={base_metrics['accuracy']:.4f} "
        f"({base_metrics['correct']}/{base_metrics['total']}), unparsed={base_metrics['unparsed']}"
    )
    print(
        f"Fine-tuned model: accuracy={tuned_metrics['accuracy']:.4f} "
        f"({tuned_metrics['correct']}/{tuned_metrics['total']}), unparsed={tuned_metrics['unparsed']}"
    )
    print(f"Delta:            accuracy={tuned_metrics['accuracy'] - base_metrics['accuracy']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
