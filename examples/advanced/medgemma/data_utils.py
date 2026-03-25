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
MedGemma task labels, prompts, and data helpers.

Adapted from the Google Health MedGemma Hugging Face fine-tuning notebook:
https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
"""

from __future__ import annotations

import os
import random
from typing import Any

DEFAULT_MODEL_NAME_OR_PATH = "google/medgemma-4b-it"
DEFAULT_EVAL_DATASET_DIR = "CRC-VAL-HE-7K"

RAW_TISSUE_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TISSUE_CLASSES = [
    "A: adipose",
    "B: background",
    "C: debris",
    "D: lymphocytes",
    "E: mucus",
    "F: smooth muscle",
    "G: normal colon mucosa",
    "H: cancer-associated stroma",
    "I: colorectal adenocarcinoma epithelium",
]

RAW_TISSUE_TO_INDEX = {label: idx for idx, label in enumerate(RAW_TISSUE_CODES)}
ALT_TISSUE_LABELS = {label: f"({label.replace(': ', ') ')}" for label in TISSUE_CLASSES}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

PROMPT = "What is the most likely tissue type shown in the histopathology image?\n" + "\n".join(TISSUE_CLASSES)


def resolve_image_path(image_path: str, image_root: str) -> str:
    path = image_path.strip()
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(image_root, path))


def format_training_example(example: dict[str, Any]) -> dict[str, Any]:
    label_name = example.get("label_name") or TISSUE_CLASSES[int(example["label"])]
    formatted = dict(example)
    formatted["label_name"] = label_name
    formatted["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": label_name},
            ],
        },
    ]
    return formatted


def format_inference_example(example: dict[str, Any]) -> dict[str, Any]:
    formatted = dict(example)
    formatted["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return formatted


def parse_prediction_label(response_text: str) -> int:
    for label_index, label_name in enumerate(TISSUE_CLASSES):
        if label_name in response_text or ALT_TISSUE_LABELS[label_name] in response_text:
            return label_index
    return -1


def sample_records(records: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if max_samples is None or max_samples <= 0:
        return shuffled
    return shuffled[:max_samples]


def collect_image_records(dataset_dir: str) -> list[dict[str, Any]]:
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records = []
    for root, dir_names, file_names in os.walk(dataset_dir):
        dir_names.sort()
        file_names.sort()
        raw_label = os.path.basename(root).upper()
        if raw_label not in RAW_TISSUE_TO_INDEX:
            continue

        label_index = RAW_TISSUE_TO_INDEX[raw_label]
        label_name = TISSUE_CLASSES[label_index]
        for file_name in file_names:
            if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, dataset_dir)
            records.append(
                {
                    "image": rel_path,
                    "label": label_index,
                    "label_name": label_name,
                    "raw_label": raw_label,
                }
            )

    if not records:
        raise FileNotFoundError(
            f"No labeled images were found under {dataset_dir}. "
            f"Expected directories named like {', '.join(RAW_TISSUE_CODES)}."
        )
    return records


def split_records_for_clients(
    records: list[dict[str, Any]],
    num_clients: int,
    samples_per_client: int | None,
    validation_size_per_client: int,
    seed: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")
    if validation_size_per_client < 0:
        raise ValueError(f"validation_size_per_client must be >= 0, got {validation_size_per_client}")
    if samples_per_client is not None and samples_per_client <= 0:
        raise ValueError(f"samples_per_client must be > 0 when set, got {samples_per_client}")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    if samples_per_client is None:
        base = len(shuffled) // num_clients
        remainder = len(shuffled) % num_clients
        counts = [base + (1 if idx < remainder else 0) for idx in range(num_clients)]
    else:
        max_samples_per_client = len(shuffled) // num_clients
        if max_samples_per_client <= 0:
            raise ValueError(
                f"Insufficient data: {len(shuffled)} total samples cannot be split across {num_clients} clients."
            )
        count = min(samples_per_client, max_samples_per_client)
        counts = [count for _ in range(num_clients)]

    site_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cursor = 0
    for client_idx, count in enumerate(counts, start=1):
        if count <= validation_size_per_client:
            raise ValueError(
                f"Client shard size {count} must be larger than validation_size_per_client={validation_size_per_client}."
            )

        shard = shuffled[cursor : cursor + count]
        cursor += count
        site_splits[f"site-{client_idx}"] = {
            "train": shard[validation_size_per_client:],
            "validation": shard[:validation_size_per_client],
        }

    return site_splits
