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
from collections import Counter
from typing import Any
from xml.sax.saxutils import escape

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
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SVG_LABEL_COLORS = [
    "#D1495B",
    "#EDA65D",
    "#F7DC6F",
    "#5DA271",
    "#2E86AB",
    "#8E6C8A",
    "#556B2F",
    "#B56576",
    "#4D908E",
]

PROMPT = "What is the most likely tissue type shown in the histopathology image?\n" + "\n".join(TISSUE_CLASSES)


def _to_alt_tissue_label(label: str) -> str:
    code, description = label.split(": ", 1)
    return f"({code}) {description}"


ALT_TISSUE_LABELS = {label: _to_alt_tissue_label(label) for label in TISSUE_CLASSES}


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


def _compute_client_sample_counts(total_records: int, num_clients: int, samples_per_client: int | None) -> list[int]:
    if samples_per_client is None:
        base = total_records // num_clients
        remainder = total_records % num_clients
        return [base + (1 if idx < remainder else 0) for idx in range(num_clients)]

    max_samples_per_client = total_records // num_clients
    if max_samples_per_client <= 0:
        raise ValueError(
            f"Insufficient data: {total_records} total samples cannot be split across {num_clients} clients."
        )
    count = min(samples_per_client, max_samples_per_client)
    return [count for _ in range(num_clients)]


def _build_dominant_label_groups(num_clients: int, num_labels: int) -> list[list[int]]:
    if num_clients <= num_labels:
        base = num_labels // num_clients
        remainder = num_labels % num_clients
        groups = []
        cursor = 0
        for client_idx in range(num_clients):
            group_size = base + (1 if client_idx < remainder else 0)
            groups.append(list(range(cursor, cursor + group_size)))
            cursor += group_size
        return groups

    return [[client_idx % num_labels] for client_idx in range(num_clients)]


def _build_label_preference_matrix(
    num_clients: int, num_labels: int, dominant_fraction: float
) -> tuple[list[list[float]], list[list[int]]]:
    if not 0.0 <= dominant_fraction <= 1.0:
        raise ValueError(f"dominant_fraction must be between 0 and 1 inclusive, got {dominant_fraction}")

    dominant_groups = _build_dominant_label_groups(num_clients, num_labels)
    preferences = []
    for dominant_labels in dominant_groups:
        if len(dominant_labels) == num_labels:
            preferences.append([1.0 / num_labels for _ in range(num_labels)])
            continue

        dominant_weight = dominant_fraction / len(dominant_labels)
        nondominant_count = num_labels - len(dominant_labels)
        nondominant_weight = (1.0 - dominant_fraction) / nondominant_count
        weights = [nondominant_weight for _ in range(num_labels)]
        for label_idx in dominant_labels:
            weights[label_idx] = dominant_weight
        preferences.append(weights)
    return preferences, dominant_groups


def _weighted_choice(rng: random.Random, candidates: list[int], weights: list[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)

    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if cumulative >= threshold:
            return candidate
    return candidates[-1]


def _split_records_random(
    records: list[dict[str, Any]],
    num_clients: int,
    samples_per_client: int | None,
    validation_size_per_client: int,
    seed: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    counts = _compute_client_sample_counts(len(shuffled), num_clients, samples_per_client)

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


def _split_records_heterogeneous(
    records: list[dict[str, Any]],
    num_clients: int,
    samples_per_client: int | None,
    validation_size_per_client: int,
    seed: int,
    dominant_fraction: float,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rng = random.Random(seed)
    selected_records = list(records)
    rng.shuffle(selected_records)

    counts = _compute_client_sample_counts(len(selected_records), num_clients, samples_per_client)
    preferences, dominant_groups = _build_label_preference_matrix(
        num_clients=num_clients,
        num_labels=len(TISSUE_CLASSES),
        dominant_fraction=dominant_fraction,
    )
    remaining_capacity = list(counts)
    assigned_by_client: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]

    for record in selected_records:
        available_clients = [idx for idx, capacity in enumerate(remaining_capacity) if capacity > 0]
        if not available_clients:
            break

        label_idx = int(record["label"])
        weights = [
            preferences[client_idx][label_idx] * remaining_capacity[client_idx] for client_idx in available_clients
        ]
        chosen_client = _weighted_choice(rng, available_clients, weights)
        assigned_by_client[chosen_client].append(record)
        remaining_capacity[chosen_client] -= 1

    assert not any(
        capacity > 0 for capacity in remaining_capacity
    ), f"BUG: Could not fill all client capacities. Remaining: {remaining_capacity}"

    site_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for client_idx, (count, assigned_records, dominant_labels) in enumerate(
        zip(counts, assigned_by_client, dominant_groups),
        start=1,
    ):
        if count <= validation_size_per_client:
            raise ValueError(
                f"Client shard size {count} must be larger than validation_size_per_client={validation_size_per_client}."
            )

        rng.shuffle(assigned_records)
        site_splits[f"site-{client_idx}"] = {
            "train": assigned_records[validation_size_per_client:],
            "validation": assigned_records[:validation_size_per_client],
            "dominant_labels": [TISSUE_CLASSES[label_idx] for label_idx in dominant_labels],
        }

    return site_splits


def summarize_label_distribution(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(int(record["label"]) for record in records)
    return [f"{TISSUE_CLASSES[label_idx]}={counts.get(label_idx, 0)}" for label_idx in range(len(TISSUE_CLASSES))]


def _label_count_vector(records: list[dict[str, Any]]) -> list[int]:
    counts = Counter(int(record["label"]) for record in records)
    return [counts.get(label_idx, 0) for label_idx in range(len(TISSUE_CLASSES))]


def write_label_distribution_svg(site_splits: dict[str, dict[str, list[dict[str, Any]]]], output_path: str) -> None:
    site_names = sorted(site_splits)
    label_count_vectors = {site_name: _label_count_vector(site_splits[site_name]["train"]) for site_name in site_names}
    max_count = max((max(counts) for counts in label_count_vectors.values()), default=1)
    max_count = max(max_count, 1)

    width = 980
    top_margin = 28
    panel_title_height = 52
    row_height = 28
    panel_gap = 22
    panel_height = panel_title_height + len(TISSUE_CLASSES) * row_height + 30
    height = top_margin + len(site_names) * panel_height + max(0, len(site_names) - 1) * panel_gap + 28

    label_x = 32
    label_width = 290
    bar_x = label_x + label_width
    bar_width = 470
    value_x = bar_x + bar_width + 18

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">MedGemma client train label distribution</title>',
        '  <desc id="desc">Per-client training-label distribution for the heterogeneous MedGemma split.</desc>',
        '  <rect width="100%" height="100%" fill="#FFFDF8"/>',
        '  <text x="32" y="26" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1F2933">'
        "Client Train Label Distribution"
        "</text>",
    ]

    for site_idx, site_name in enumerate(site_names):
        split_data = site_splits[site_name]
        dominant_labels = split_data.get("dominant_labels", [])
        counts = label_count_vectors[site_name]
        panel_top = top_margin + site_idx * (panel_height + panel_gap)

        lines.extend(
            [
                f'  <rect x="20" y="{panel_top}" width="{width - 40}" height="{panel_height}" rx="14" fill="#FFFFFF" stroke="#D8DEE9"/>',
                f'  <text x="36" y="{panel_top + 26}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#102A43">{escape(site_name)}</text>',
            ]
        )
        if dominant_labels:
            dominant_text = "dominant: " + ", ".join(dominant_labels)
            lines.append(
                f'  <text x="150" y="{panel_top + 26}" font-family="Arial, sans-serif" font-size="12" fill="#486581">{escape(dominant_text)}</text>'
            )

        for label_idx, label_name in enumerate(TISSUE_CLASSES):
            row_top = panel_top + panel_title_height + label_idx * row_height
            bar_length = 0 if max_count <= 0 else counts[label_idx] / max_count * bar_width
            bar_color = SVG_LABEL_COLORS[label_idx % len(SVG_LABEL_COLORS)]
            lines.extend(
                [
                    f'  <text x="{label_x}" y="{row_top + 15}" font-family="Arial, sans-serif" font-size="12" fill="#243B53">{escape(label_name)}</text>',
                    f'  <rect x="{bar_x}" y="{row_top + 4}" width="{bar_width}" height="14" rx="7" fill="#E5E7EB"/>',
                    f'  <rect x="{bar_x}" y="{row_top + 4}" width="{bar_length:.1f}" height="14" rx="7" fill="{bar_color}"/>',
                    f'  <text x="{value_x}" y="{row_top + 15}" font-family="Arial, sans-serif" font-size="12" fill="#243B53">{counts[label_idx]}</text>',
                ]
            )

    lines.append("</svg>")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def split_records_for_clients(
    records: list[dict[str, Any]],
    num_clients: int,
    samples_per_client: int | None,
    validation_size_per_client: int,
    seed: int,
    split_strategy: str = "heterogeneous",
    dominant_fraction: float = 0.8,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")
    if validation_size_per_client < 0:
        raise ValueError(f"validation_size_per_client must be >= 0, got {validation_size_per_client}")
    if samples_per_client is not None and samples_per_client <= 0:
        raise ValueError(f"samples_per_client must be > 0 when set, got {samples_per_client}")
    if split_strategy == "random":
        return _split_records_random(
            records=records,
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            validation_size_per_client=validation_size_per_client,
            seed=seed,
        )
    if split_strategy == "heterogeneous":
        return _split_records_heterogeneous(
            records=records,
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            validation_size_per_client=validation_size_per_client,
            seed=seed,
            dominant_fraction=dominant_fraction,
        )
    raise ValueError(f"Unsupported split_strategy {split_strategy!r}. Use 'heterogeneous' or 'random'.")
