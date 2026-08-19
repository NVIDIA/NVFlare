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

"""Deterministic synthetic multimodal data for the FedCoRe starter."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

SCHEMA_VERSION = 1
SITE_IMAGE_FRACTIONS = (1.0, 0.5, 0.0)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SyntheticDataConfig:
    output_dir: Path
    train_samples_per_site: int = 48
    val_samples_per_site: int = 16
    test_samples_per_site: int = 16
    proxy_strength: float = 0.9
    seed: int = 7
    image_size: int = 224

    def split_count(self, split: str) -> int:
        return {
            "train": self.train_samples_per_site,
            "val": self.val_samples_per_site,
            "test": self.test_samples_per_site,
        }[split]


def make_question(context: str, *, include_image: bool) -> str:
    image_status = (
        "Use the image as the authoritative signal."
        if include_image
        else "The image is unavailable; use only the auxiliary context."
    )
    return (
        "This is a synthetic classification task. A red triangle is class A and a blue circle is class B. "
        f"{image_status} Auxiliary context: {context} Return exactly A or B."
    )


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def _draw_marker(path: Path, label: int, image_size: int) -> None:
    image = Image.new("RGB", (image_size, image_size), color=(238, 240, 244))
    draw = ImageDraw.Draw(image)
    margin = max(20, image_size // 5)
    if label == 1:
        points = [
            (image_size // 2, margin),
            (image_size - margin, image_size - margin),
            (margin, image_size - margin),
        ]
        draw.polygon(points, fill=(214, 39, 40), outline=(105, 20, 20), width=4)
    else:
        draw.ellipse(
            (margin, margin, image_size - margin, image_size - margin),
            fill=(31, 119, 180),
            outline=(12, 55, 85),
            width=4,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _balanced_labels(count: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(count, dtype=np.int64) % 2
    rng.shuffle(labels)
    return labels


def _stratified_mask(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Select the requested fraction independently within each class."""
    mask = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        selected = int(round(len(indices) * fraction))
        if selected:
            mask[rng.permutation(indices)[:selected]] = True
    return mask


def _context(label: int, proxy_matches: bool, rng: np.random.Generator) -> tuple[str, int]:
    proxy_label = label if proxy_matches else 1 - label
    code = "KAPPA" if proxy_label else "SIGMA"
    batch = int(rng.integers(100, 1000))
    return f"auxiliary scanner code={code}; acquisition batch={batch}.", proxy_label


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _sft_record(record: dict) -> dict:
    question = make_question(record["context"], include_image=record["image_available"])
    if record["image_available"]:
        question = f"<image>\n{question}"
    result = {
        "id": record["example_id"],
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": "A" if record["label"] else "B"},
        ],
    }
    if record["image_available"]:
        result["image"] = [record["image"]]
    return result


def generate_synthetic_data(config: SyntheticDataConfig) -> dict:
    if not 0.0 <= config.proxy_strength <= 1.0:
        raise ValueError("proxy_strength must be in [0, 1].")
    if min(config.split_count(split) for split in SPLITS) < 2:
        raise ValueError("Each site split must contain at least two examples.")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "proxy_strength": config.proxy_strength,
        "sites": {},
        "splits": {},
    }
    all_ids = set()

    for site_index, image_fraction in enumerate(SITE_IMAGE_FRACTIONS, start=1):
        site_name = f"site-{site_index}"
        site_dir = output_dir / site_name
        site_summary = {"image_fraction": image_fraction, "splits": {}}
        train_sft = []

        for split_index, split in enumerate(SPLITS):
            count = config.split_count(split)
            rng = np.random.default_rng(config.seed + 1000 * site_index + 100 * split_index)
            labels = _balanced_labels(count, rng)
            availability = _stratified_mask(labels, image_fraction, rng)
            proxy_matches = _stratified_mask(labels, config.proxy_strength, rng)
            records = []
            for local_index, (label, image_available, proxy_matches_label) in enumerate(
                zip(labels, availability, proxy_matches)
            ):
                example_id = f"{split}-s{site_index}-{local_index:05d}"
                if example_id in all_ids:
                    raise RuntimeError(f"Duplicate example ID generated: {example_id}")
                all_ids.add(example_id)
                context, proxy_label = _context(int(label), bool(proxy_matches_label), rng)
                image_rel = f"{site_name}/images/{example_id}.png" if image_available else ""
                if image_available:
                    _draw_marker(output_dir / image_rel, int(label), config.image_size)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": example_id,
                    "site": site_name,
                    "split": split,
                    "label": int(label),
                    "answer": "A" if int(label) else "B",
                    "context": context,
                    "proxy_label": int(proxy_label),
                    "proxy_matches_label": bool(proxy_matches_label),
                    "image_available": bool(image_available),
                    "image": image_rel,
                }
                records.append(record)
                if split == "train":
                    train_sft.append(_sft_record(record))

            manifest_path = site_dir / f"{split}.jsonl"
            _write_jsonl(manifest_path, records)
            split_summary = {
                "examples": count,
                "positive": int(labels.sum()),
                "image_available": int(availability.sum()),
                "image_missing": int((~availability).sum()),
                "proxy_matches_label": int(proxy_matches.sum()),
            }
            site_summary["splits"][split] = split_summary
            summary["splits"].setdefault(split, 0)
            summary["splits"][split] += count

        with (site_dir / "train.json").open("w") as f:
            json.dump(train_sft, f, indent=2)
        summary["sites"][site_name] = site_summary

    summary["total_examples"] = len(all_ids)
    with (output_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary
