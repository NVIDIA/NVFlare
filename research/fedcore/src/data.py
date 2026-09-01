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

"""Deterministic MNIST image-plus-context data for the FedCoRe starter."""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SCHEMA_VERSION = 2
SITE_IMAGE_FRACTIONS = (1.0, 0.5, 0.0)
SPLITS = ("train", "val", "test")
MNISTLoader = Callable[[Path, bool], object]


@dataclass(frozen=True)
class MNISTDataConfig:
    output_dir: Path
    dataset_root: Path
    scenario: str = "recoverable"
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
        "Use the handwritten-digit image as the authoritative signal."
        if include_image
        else "The handwritten-digit image is unavailable; use only the secondary OCR report."
    )
    return (
        "This is an MNIST digit classification task. Class A contains digits 0 through 4 and class B contains digits "
        f"5 through 9. {image_status} Secondary OCR report: {context} Return exactly A or B."
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


def _load_torchvision_mnist(root: Path, train: bool):
    from torchvision.datasets import MNIST

    return MNIST(root=str(root), train=train, download=True)


def _binary_label(digit: int) -> int:
    return int(digit <= 4)


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        values = dataset.targets
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.int64)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))], dtype=np.int64)


def _allocate_balanced_indices(dataset, requests: list[tuple[str, int]], seed: int) -> dict[str, list[int]]:
    targets = _targets(dataset)
    rng = np.random.default_rng(seed)
    pools = {}
    for label in (0, 1):
        indices = np.flatnonzero(np.asarray([_binary_label(int(digit)) for digit in targets]) == label)
        pools[label] = rng.permutation(indices)
    positions = {0: 0, 1: 0}
    allocations = {}
    for key, count in requests:
        if count % 2:
            raise ValueError(f"MNIST split size for {key} must be even, got {count}.")
        selected = []
        per_class = count // 2
        for label in (0, 1):
            start = positions[label]
            end = start + per_class
            if end > len(pools[label]):
                raise ValueError(f"Not enough MNIST examples to allocate {key}.")
            selected.extend(int(index) for index in pools[label][start:end])
            positions[label] = end
        allocations[key] = [int(index) for index in rng.permutation(selected)]
    return allocations


def _stratified_mask(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Select the requested fraction independently within each class."""

    mask = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        selected = int(np.floor(len(indices) * fraction + 0.5))
        if selected:
            mask[rng.permutation(indices)[:selected]] = True
    return mask


def _confidence_mask(
    labels: np.ndarray,
    sensor_matches: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    high_confidence = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        for matches in (False, True):
            indices = np.flatnonzero((labels == label) & (sensor_matches == matches))
            if not len(indices):
                continue
            if scenario == "recoverable":
                fraction = 0.8 if matches else 0.2
            else:
                fraction = 0.5
            selected = int(np.floor(len(indices) * fraction + 0.5))
            if selected:
                high_confidence[rng.permutation(indices)[:selected]] = True
    return high_confidence


def _ocr_digit(true_digit: int, sensor_matches: bool, rng: np.random.Generator) -> int:
    true_label = _binary_label(true_digit)
    estimated_label = true_label if sensor_matches else 1 - true_label
    candidates = list(range(0, 5)) if estimated_label == 1 else list(range(5, 10))
    if sensor_matches and rng.random() < 0.7:
        return true_digit
    alternatives = [digit for digit in candidates if digit != true_digit]
    return int(rng.choice(alternatives or candidates))


def _render_mnist_image(source: Image.Image, path: Path, image_size: int) -> None:
    if image_size < 64:
        raise ValueError("image_size must be at least 64 pixels.")
    digit = ImageOps.invert(source.convert("L"))
    marker_size = int(round(image_size * 0.78))
    digit = digit.resize((marker_size, marker_size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (image_size, image_size), color=255)
    offset = (image_size - marker_size) // 2
    canvas.paste(digit, (offset, offset))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path)


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


def _validate_config(config: MNISTDataConfig) -> None:
    if config.scenario not in {"recoverable", "uninformative"}:
        raise ValueError("scenario must be 'recoverable' or 'uninformative'.")
    if not 0.0 <= config.proxy_strength <= 1.0:
        raise ValueError("proxy_strength must be in [0, 1].")
    for split in SPLITS:
        count = config.split_count(split)
        if count < 8 or count % 4:
            raise ValueError(f"{split}_samples_per_site must be at least 8 and divisible by 4, got {count}.")
        if config.scenario == "uninformative" and count % 8:
            raise ValueError(
                f"{split}_samples_per_site must be divisible by 8 for the exactly balanced control, got {count}."
            )


def generate_mnist_data(config: MNISTDataConfig, dataset_loader: MNISTLoader | None = None) -> dict:
    _validate_config(config)
    output_dir = config.output_dir.expanduser().resolve()
    dataset_root = config.dataset_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = dataset_loader or _load_torchvision_mnist
    train_dataset = loader(dataset_root, True)
    test_dataset = loader(dataset_root, False)

    train_requests = [
        (f"{split}:site-{site_index}", config.split_count(split))
        for split in ("train", "val")
        for site_index in range(1, 4)
    ]
    test_requests = [(f"test:site-{site_index}", config.test_samples_per_site) for site_index in range(1, 4)]
    allocations = _allocate_balanced_indices(train_dataset, train_requests, seed=config.seed + 17)
    allocations.update(_allocate_balanced_indices(test_dataset, test_requests, seed=config.seed + 29))

    effective_proxy_strength = config.proxy_strength if config.scenario == "recoverable" else 0.5
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "MNIST",
        "dataset_root": str(dataset_root),
        "scenario": config.scenario,
        "seed": config.seed,
        "proxy_strength": effective_proxy_strength,
        "sites": {},
        "splits": {},
    }
    all_source_ids = set()

    for site_index, image_fraction in enumerate(SITE_IMAGE_FRACTIONS, start=1):
        site_name = f"site-{site_index}"
        site_dir = output_dir / site_name
        site_summary = {"image_fraction": image_fraction, "splits": {}}
        train_sft = []

        for split_index, split in enumerate(SPLITS):
            dataset = test_dataset if split == "test" else train_dataset
            source_split = "test" if split == "test" else "train"
            indices = allocations[f"{split}:{site_name}"]
            digits = np.asarray([int(dataset[index][1]) for index in indices], dtype=np.int64)
            labels = np.asarray([_binary_label(int(digit)) for digit in digits], dtype=np.int64)
            rng = np.random.default_rng(config.seed + 1000 * site_index + 100 * split_index)
            availability = _stratified_mask(labels, image_fraction, rng)
            sensor_matches = _stratified_mask(labels, effective_proxy_strength, rng)
            high_confidence = _confidence_mask(labels, sensor_matches, config.scenario, rng)
            records = []

            for local_index, source_index in enumerate(indices):
                source_id = f"{source_split}:{source_index}"
                if source_id in all_source_ids:
                    raise RuntimeError(f"MNIST source example was allocated more than once: {source_id}")
                all_source_ids.add(source_id)
                true_digit = int(digits[local_index])
                label = int(labels[local_index])
                matches = bool(sensor_matches[local_index])
                ocr_digit = _ocr_digit(true_digit, matches, rng)
                confidence = "high" if bool(high_confidence[local_index]) else "low"
                context = f"estimated digit={ocr_digit}; sensor confidence={confidence}."
                example_id = f"{split}-s{site_index}-{local_index:05d}"
                image_available = bool(availability[local_index])
                image_rel = f"{site_name}/images/{example_id}.png" if image_available else ""
                if image_available:
                    source_image, _ = dataset[source_index]
                    _render_mnist_image(source_image, output_dir / image_rel, config.image_size)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": "MNIST",
                    "example_id": example_id,
                    "site": site_name,
                    "split": split,
                    "source_split": source_split,
                    "source_index": int(source_index),
                    "digit": true_digit,
                    "label": label,
                    "answer": "A" if label else "B",
                    "context": context,
                    "ocr_digit": ocr_digit,
                    "ocr_label": _binary_label(ocr_digit),
                    "sensor_confidence": confidence,
                    "sensor_matches_label": matches,
                    "image_available": image_available,
                    "image": image_rel,
                }
                records.append(record)
                if split == "train":
                    train_sft.append(_sft_record(record))

            manifest_path = site_dir / f"{split}.jsonl"
            _write_jsonl(manifest_path, records)
            split_summary = {
                "examples": len(records),
                "class_a": int(labels.sum()),
                "image_available": int(availability.sum()),
                "image_missing": int((~availability).sum()),
                "sensor_matches_label": int(sensor_matches.sum()),
                "sensor_high_confidence": int(high_confidence.sum()),
            }
            site_summary["splits"][split] = split_summary
            summary["splits"].setdefault(split, 0)
            summary["splits"][split] += len(records)

        with (site_dir / "train.json").open("w") as f:
            json.dump(train_sft, f, indent=2)
        summary["sites"][site_name] = site_summary

    summary["total_examples"] = len(all_source_ids)
    with (output_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary
