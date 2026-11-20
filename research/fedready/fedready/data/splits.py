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

"""Bounded sample selection and the shared train/validation/test split policy."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Callable

from fedready.data.contracts import STANDARD_SPLITS
from fedready.data.extraction_artifacts import ExtractionConfig

COMMON_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _organize_adapter_pairs_for_fl_splits(
    pairs: list[dict[str, Any]],
    policy: dict[str, Any],
    config: ExtractionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments, plan = _assign_fl_splits(
        item_count=len(pairs),
        current_split=lambda index: str(pairs[index].get("split", "unknown")),
        stable_key=lambda index: str(pairs[index].get("stable_key", index)),
        config=config,
    )
    adjusted = [{**pair, "split": assignments[index]} for index, pair in enumerate(pairs)]
    split_rule = policy.get("split_rule", {})
    if isinstance(split_rule, dict):
        plan["policy_split_rule"] = {
            "preserve_existing_splits": bool(split_rule.get("preserve_existing_splits", True)),
            "if_validation_missing": split_rule.get("if_validation_missing"),
        }
    plan["source"] = "client_local_adapter_manifest"
    return adjusted, plan


def _bounded_sample_pairs_for_default_fl_splits(
    pairs: list[dict[str, Any]],
    policy: dict[str, Any],
    config: ExtractionConfig,
    *,
    stable_key: Callable[[dict[str, Any]], str],
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded subset and apply the default FL train/val/test split.

    Bounded benchmark runs should not inherit source train/test-only splits. They
    compare a fixed-size sample from each site, so the fixed-size subset gets the
    common 80/10/10 train/validation/test split regardless of source folder names.
    """

    max_samples = config.max_samples if config.max_samples is not None else len(pairs)
    selected_indices = _bounded_sample_indices(
        item_count=len(pairs),
        max_samples=max_samples,
        stable_key=lambda index: stable_key(pairs[index]),
        config=config,
    )
    selected = [pairs[index] for index in selected_indices]
    assignments: list[str | None] = [None] * len(selected)
    _assign_common_ratio(
        indices=list(range(len(selected))),
        assignments=assignments,
        stable_key=lambda index: stable_key(selected[index]),
        config=config,
    )
    final_assignments = [split if split in STANDARD_SPLITS else "train" for split in assignments]
    adjusted = [{**pair, "split": final_assignments[index]} for index, pair in enumerate(selected)]
    input_counts = Counter(
        str(pair.get("split", "unknown")) if pair.get("split") in STANDARD_SPLITS else "unknown" for pair in pairs
    )
    output_counts = Counter(final_assignments)
    plan = {
        "schema_version": "fedready.extracted_split_plan.v1",
        "strategy": "bounded_sample_common_ratio_split",
        "official_split_detected": False,
        "source": source,
        "default_ratio": COMMON_SPLIT_RATIOS,
        "seed": config.split_seed,
        "bounded_sample_split": True,
        "max_samples": max_samples,
        "full_valid_case_count": len(pairs),
        "selected_case_count": len(selected),
        "created_splits": sorted({split for split in final_assignments}),
        "input_counts": {split: input_counts.get(split, 0) for split in (*STANDARD_SPLITS, "unknown")},
        "output_counts": {split: output_counts.get(split, 0) for split in STANDARD_SPLITS},
        "safe_to_share": True,
    }
    split_rule = policy.get("split_rule", {})
    if isinstance(split_rule, dict):
        plan["policy_split_rule"] = {
            "preserve_existing_splits": bool(split_rule.get("preserve_existing_splits", True)),
            "if_validation_missing": split_rule.get("if_validation_missing"),
            "overridden_for_bounded_sample": True,
        }
    return adjusted, plan


def _bounded_sample_indices(
    *,
    item_count: int,
    max_samples: int,
    stable_key: Callable[[int], str],
    config: ExtractionConfig,
) -> list[int]:
    indices = list(range(item_count))
    if max_samples >= item_count:
        return indices
    selected = sorted(indices, key=lambda index: _split_hash(config.split_seed, f"bounded:{stable_key(index)}"))[
        :max_samples
    ]
    return sorted(selected)


def _assign_fl_splits(
    *,
    item_count: int,
    current_split: Any,
    stable_key: Any,
    config: ExtractionConfig,
) -> tuple[list[str], dict[str, Any]]:
    raw_splits = [current_split(index) for index in range(item_count)]
    input_counts = Counter(split if split in STANDARD_SPLITS else "unknown" for split in raw_splits)
    known_splits = {split for split in STANDARD_SPLITS if input_counts[split] > 0}
    official_split_detected = len(known_splits) >= 2 or bool(known_splits & {"validation", "test"})

    assignments: list[str | None] = [None] * item_count
    strategy = "common_ratio_split"
    created_splits: list[str] = []

    if official_split_detected:
        strategy = "preserve_official_splits"
        unknown_indices: list[int] = []
        for index, split in enumerate(raw_splits):
            if split in STANDARD_SPLITS:
                assignments[index] = split
            else:
                unknown_indices.append(index)
        if unknown_indices:
            strategy = "preserve_official_splits_with_common_ratio_for_unknown"
            created_splits.append("unknown_to_standard")
            _assign_common_ratio(indices=unknown_indices, assignments=assignments, stable_key=stable_key, config=config)
    else:
        _assign_common_ratio(
            indices=list(range(item_count)), assignments=assignments, stable_key=stable_key, config=config
        )
        created_splits.append("all_cases")

    if item_count:
        if (
            "validation" not in {split for split in assignments if split in STANDARD_SPLITS}
            and config.validation_fraction > 0
        ):
            if _move_train_subset(
                assignments=assignments,
                stable_key=stable_key,
                config=config,
                target_split="validation",
                fraction=config.validation_fraction,
            ):
                created_splits.append("validation")
        if "test" not in {split for split in assignments if split in STANDARD_SPLITS}:
            if _move_train_subset(
                assignments=assignments,
                stable_key=stable_key,
                config=config,
                target_split="test",
                fraction=COMMON_SPLIT_RATIOS["test"],
            ):
                created_splits.append("test")

    final_assignments = [split if split in STANDARD_SPLITS else "train" for split in assignments]
    output_counts = Counter(final_assignments)
    plan = {
        "schema_version": "fedready.extracted_split_plan.v1",
        "strategy": strategy,
        "official_split_detected": official_split_detected,
        "source": "source_path_tokens" if official_split_detected else "common_ratio_80_10_10",
        "default_ratio": COMMON_SPLIT_RATIOS,
        "seed": config.split_seed,
        "created_splits": sorted(set(created_splits)),
        "input_counts": {split: input_counts.get(split, 0) for split in (*STANDARD_SPLITS, "unknown")},
        "output_counts": {split: output_counts.get(split, 0) for split in STANDARD_SPLITS},
        "safe_to_share": True,
    }
    return final_assignments, plan


def _assign_common_ratio(
    *,
    indices: list[int],
    assignments: list[str | None],
    stable_key: Any,
    config: ExtractionConfig,
) -> None:
    counts = _common_split_counts(len(indices))
    ordered = sorted(indices, key=lambda index: _split_hash(config.split_seed, stable_key(index)))
    validation_end = counts["validation"]
    test_end = validation_end + counts["test"]
    for offset, index in enumerate(ordered):
        if offset < validation_end:
            assignments[index] = "validation"
        elif offset < test_end:
            assignments[index] = "test"
        else:
            assignments[index] = "train"


def _common_split_counts(total: int) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if total < 3:
        return {"train": total, "validation": 0, "test": 0}
    validation = max(1, round(total * COMMON_SPLIT_RATIOS["validation"]))
    test = max(1, round(total * COMMON_SPLIT_RATIOS["test"]))
    if validation + test >= total:
        validation = 1
        test = 1 if total > 2 else 0
    return {"train": total - validation - test, "validation": validation, "test": test}


def _move_train_subset(
    *,
    assignments: list[str | None],
    stable_key: Any,
    config: ExtractionConfig,
    target_split: str,
    fraction: float,
) -> bool:
    train_indices = [index for index, split in enumerate(assignments) if split == "train"]
    if len(train_indices) < 2:
        return False
    move_count = max(1, round(len(train_indices) * fraction))
    move_count = min(move_count, len(train_indices) - 1)
    ordered = sorted(
        train_indices, key=lambda index: _split_hash(config.split_seed, f"{target_split}:{stable_key(index)}")
    )
    for index in ordered[:move_count]:
        assignments[index] = target_split
    return move_count > 0


def _split_hash(seed: int, key: str) -> str:
    return sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
