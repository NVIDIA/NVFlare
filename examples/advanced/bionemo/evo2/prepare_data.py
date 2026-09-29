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

"""Prepare the Nucleotide Transformer splice-site task for federated Evo2 fine-tuning."""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

DATASET_ID = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
DATASET_REVISION = "851f9946252e90c665cdb3cc3eedb78f1f26197c"
TASK = "splice_sites_all"
SOURCE_FILES = {
    "train": "splice_sites_all/train.parquet",
    "test": "splice_sites_all/test.parquet",
}
SOURCE_SPLIT_COUNTS = {"train": 30_000, "test": 3_000}
SOURCE_SEQUENCE_LENGTH = 600
SOURCE_LABELS = frozenset({0, 1, 2})
OUTPUT_FIELDS = ("sequence", "name", "label", "task")

_INTERVAL_PATTERN = re.compile(r"^(?P<chromosome>[^:\s]+):(?P<start>[0-9]+)-(?P<end>[0-9]+)(?:\|.*)?$")


class DataLeakageError(ValueError):
    """Raised when two output splits contain genomic leakage."""


class _DisjointSet:
    def __init__(self, size):
        self.parents = list(range(size))
        self.ranks = [0] * size

    def find(self, item):
        parent = self.parents[item]
        if parent != item:
            self.parents[item] = self.find(parent)
        return self.parents[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.ranks[left_root] < self.ranks[right_root]:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root
        if self.ranks[left_root] == self.ranks[right_root]:
            self.ranks[left_root] += 1


def _label_sort_key(label):
    return type(label).__name__, repr(label)


def _canonical_sequence(record):
    return "".join(record["sequence"].split()).upper()


def _label_histogram(records):
    counts = Counter(record["label"] for record in records)
    return {str(label): counts[label] for label in sorted(counts, key=_label_sort_key)}


def _normalize_record(record):
    missing = [field for field in OUTPUT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Dataset record is missing required fields: {', '.join(missing)}")

    sequence = record["sequence"]
    name = record["name"]
    label = record["label"]
    task = record["task"]
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Dataset record 'sequence' must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset record 'name' must be a non-empty string")
    if task != TASK:
        raise ValueError(f"Expected task {TASK!r}, but record {name!r} has task {task!r}")
    try:
        hash(label)
        json.dumps(label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {name!r} has a non-serializable label") from exc

    return {"sequence": sequence.strip(), "name": name.strip(), "label": label, "task": task}


def _normalize_records(records):
    return [_normalize_record(record) for record in records]


def _validate_source_contract(records_by_split, expected_counts=None, sequence_length=None, labels=None):
    """Validate the pinned benchmark shape before deriving any output split."""

    expected_counts = SOURCE_SPLIT_COUNTS if expected_counts is None else expected_counts
    sequence_length = SOURCE_SEQUENCE_LENGTH if sequence_length is None else sequence_length
    labels = SOURCE_LABELS if labels is None else frozenset(labels)
    if set(records_by_split) != set(expected_counts):
        raise ValueError(
            "Downloaded dataset splits do not match the pinned source contract: "
            f"expected {sorted(expected_counts)}, observed {sorted(records_by_split)}."
        )

    observed_labels = set()
    for split_name in expected_counts:
        records = records_by_split[split_name]
        expected_count = expected_counts[split_name]
        if len(records) != expected_count:
            raise ValueError(
                f"Dataset split {split_name!r} does not match the pinned source contract: "
                f"expected {expected_count} {TASK!r} records, observed {len(records)}."
            )
        for row_index, record in enumerate(records):
            sequence = _canonical_sequence(record)
            if len(sequence) != sequence_length:
                raise ValueError(
                    f"Dataset split {split_name!r} row {row_index} ({record['name']!r}) does not match the "
                    f"pinned source contract: expected {sequence_length} bases, observed {len(sequence)}."
                )
            label = record["label"]
            if type(label) is not int or label not in labels:
                raise ValueError(
                    f"Dataset split {split_name!r} row {row_index} ({record['name']!r}) has label {label!r}; "
                    f"the pinned source contract requires integer labels {sorted(labels)}."
                )
            observed_labels.add(label)
    if observed_labels != labels:
        raise ValueError(
            "Downloaded dataset labels do not match the pinned source contract: "
            f"expected {sorted(labels)}, observed {sorted(observed_labels)}."
        )


def parse_genomic_interval(name):
    """Parse the chromosome and half-open interval encoded by a dataset record name."""

    match = _INTERVAL_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Record name does not contain a genomic interval: {name!r}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end <= start:
        raise ValueError(f"Record name has an invalid genomic interval: {name!r}")
    return match.group("chromosome"), start, end


def _build_leakage_groups(records):
    """Group records connected by an exact sequence match or overlapping interval."""

    disjoint_set = _DisjointSet(len(records))
    first_sequence_index = {}
    intervals = []
    for index, record in enumerate(records):
        sequence = _canonical_sequence(record)
        previous_index = first_sequence_index.setdefault(sequence, index)
        disjoint_set.union(index, previous_index)
        chromosome, start, end = parse_genomic_interval(record["name"])
        intervals.append((chromosome, start, end, index))

    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    active_chromosome = None
    active_end = -1
    active_index = -1
    for chromosome, start, end, index in intervals:
        if chromosome != active_chromosome or start >= active_end:
            active_chromosome = chromosome
            active_end = end
            active_index = index
            continue
        disjoint_set.union(index, active_index)
        if end > active_end:
            active_end = end
            active_index = index

    grouped_indices = defaultdict(list)
    for index in range(len(records)):
        grouped_indices[disjoint_set.find(index)].append(index)
    return sorted(grouped_indices.values(), key=lambda group: group[0])


def _proportional_counts(label_counts, total_to_select):
    total = sum(label_counts.values())
    if not 0 <= total_to_select <= total:
        raise ValueError(f"Cannot select {total_to_select} records from {total}")
    if total == 0:
        return {}

    quotas = {}
    remainders = []
    for label in sorted(label_counts, key=_label_sort_key):
        exact_count = label_counts[label] * total_to_select / total
        base_count = math.floor(exact_count)
        quotas[label] = base_count
        remainders.append((exact_count - base_count, label))

    remaining = total_to_select - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], _label_sort_key(item[1])))
    for _, label in remainders:
        if remaining == 0:
            break
        if quotas[label] < label_counts[label]:
            quotas[label] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Unable to apportion the requested sample count across labels")
    return quotas


def _group_signature(group, records):
    label_counts = Counter(records[index]["label"] for index in group)
    return tuple(sorted(label_counts.items(), key=lambda item: _label_sort_key(item[0])))


def stratified_train_validation_split(records, validation_fraction=0.1, seed=42):
    """Create a deterministic, leakage-safe stratified validation split.

    Exact sequence duplicates and overlapping genomic windows are assigned as one
    component so that related records cannot straddle training and validation.
    """

    records = _normalize_records(records)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required to create a validation split")

    validation_size = round(len(records) * validation_fraction)
    if not 0 < validation_size < len(records):
        raise ValueError(f"validation_fraction={validation_fraction} produces an empty training or validation split")
    label_counts = Counter(record["label"] for record in records)
    validation_targets = _proportional_counts(label_counts, validation_size)

    groups = _build_leakage_groups(records)
    singleton_groups = defaultdict(list)
    multi_group_buckets = defaultdict(list)
    for group in groups:
        signature = _group_signature(group, records)
        if len(group) == 1:
            singleton_groups[signature[0][0]].append(group)
        else:
            multi_group_buckets[signature].append(group)

    rng = random.Random(seed)
    selected_indices = set()
    selected_label_counts = Counter()
    for signature in sorted(multi_group_buckets, key=repr):
        bucket = multi_group_buckets[signature]
        rng.shuffle(bucket)
        requested_group_count = math.floor(len(bucket) * validation_fraction)
        for group in bucket[:requested_group_count]:
            group_counts = Counter(records[index]["label"] for index in group)
            if all(
                selected_label_counts[label] + count <= validation_targets.get(label, 0)
                for label, count in group_counts.items()
            ):
                selected_indices.update(group)
                selected_label_counts.update(group_counts)

    for label in sorted(validation_targets, key=_label_sort_key):
        available_groups = singleton_groups[label]
        rng.shuffle(available_groups)
        needed = validation_targets[label] - selected_label_counts[label]
        if needed > len(available_groups):
            raise ValueError(
                "Cannot create the requested exact stratified validation split without genomic leakage: "
                f"label {label!r} needs {needed} independent records, but only {len(available_groups)} are available"
            )
        for group in available_groups[:needed]:
            selected_indices.add(group[0])

    if len(selected_indices) != validation_size:
        raise RuntimeError(
            f"Internal split error: selected {len(selected_indices)} validation records instead of {validation_size}"
        )
    train_records = [record for index, record in enumerate(records) if index not in selected_indices]
    validation_records = [record for index, record in enumerate(records) if index in selected_indices]
    return train_records, validation_records


def stratified_sample(records, sample_size, seed=42):
    """Select an exact-size deterministic stratified subset."""

    records = _normalize_records(records)
    if sample_size is None:
        return records
    if not 0 < sample_size <= len(records):
        raise ValueError(f"sample_size must be between 1 and {len(records)}, got {sample_size}")
    if sample_size == len(records):
        return records

    indices_by_label = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_label[record["label"]].append(index)
    targets = _proportional_counts({label: len(indices) for label, indices in indices_by_label.items()}, sample_size)
    rng = random.Random(seed)
    selected_indices = set()
    for label in sorted(indices_by_label, key=_label_sort_key):
        indices = indices_by_label[label]
        rng.shuffle(indices)
        selected_indices.update(indices[: targets[label]])
    return [record for index, record in enumerate(records) if index in selected_indices]


def _iid_target_counts(records, num_sites):
    targets = [Counter() for _ in range(num_sites)]
    label_counts = Counter(record["label"] for record in records)
    offset = 0
    for label in sorted(label_counts, key=_label_sort_key):
        for index in range(label_counts[label]):
            targets[(offset + index) % num_sites][label] += 1
        offset = (offset + label_counts[label]) % num_sites
    return targets


def _weighted_counts(total, weights):
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        weights = [1.0] * len(weights)
        weight_sum = len(weights)
    exact_counts = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(count) for count in exact_counts]
    remaining = total - sum(counts)
    ranked_indices = sorted(range(len(weights)), key=lambda index: (-(exact_counts[index] - counts[index]), index))
    for index in ranked_indices[:remaining]:
        counts[index] += 1
    return counts


def _dirichlet_target_counts(records, num_sites, alpha, seed):
    if alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be greater than zero")

    targets = [Counter() for _ in range(num_sites)]
    label_counts = Counter(record["label"] for record in records)
    rng = random.Random(seed)
    for label in sorted(label_counts, key=_label_sort_key):
        # Keep the RNG sequence compatible with the former record-level implementation.
        shuffled_indices = list(range(label_counts[label]))
        rng.shuffle(shuffled_indices)
        weights = [rng.gammavariate(alpha, 1.0) for _ in range(num_sites)]
        site_counts = _weighted_counts(label_counts[label], weights)
        for site_index, count in enumerate(site_counts):
            targets[site_index][label] = count

    for empty_index, target in enumerate(targets):
        if sum(target.values()):
            continue
        donor_index = max(range(num_sites), key=lambda index: sum(targets[index].values()))
        if sum(targets[donor_index].values()) <= 1:
            raise ValueError("Unable to give every site at least one training record")
        donor_label = max(targets[donor_index], key=lambda label: (targets[donor_index][label], _label_sort_key(label)))
        targets[donor_index][donor_label] -= 1
        targets[empty_index][donor_label] += 1
    return targets


def _assignment_cost(site_index, group_counts, assigned_counts, target_counts):
    cost = 0
    for label, count in group_counts.items():
        before = assigned_counts[site_index][label] - target_counts[site_index][label]
        after = before + count
        cost += after * after - before * before
    return cost


def _group_aware_partition(records, num_sites, target_counts, seed):
    groups = _build_leakage_groups(records)
    multi_groups = []
    singleton_groups = defaultdict(list)
    for group in groups:
        group_counts = Counter(records[index]["label"] for index in group)
        if len(group) == 1:
            singleton_groups[records[group[0]]["label"]].append((group, group_counts))
        else:
            multi_groups.append((group, group_counts))

    assigned_counts = [Counter() for _ in range(num_sites)]
    assigned_groups = [[] for _ in range(num_sites)]
    rng = random.Random(seed)
    rng.shuffle(multi_groups)
    multi_groups.sort(key=lambda item: len(item[0]), reverse=True)

    def assign_group(group, group_counts):
        candidate_sites = list(range(num_sites))
        rng.shuffle(candidate_sites)
        site_index = min(
            candidate_sites,
            key=lambda index: _assignment_cost(
                index,
                group_counts,
                assigned_counts,
                target_counts,
            ),
        )
        assigned_groups[site_index].append(group)
        assigned_counts[site_index].update(group_counts)

    for group, group_counts in multi_groups:
        assign_group(group, group_counts)
    for label in sorted(singleton_groups, key=_label_sort_key):
        groups_for_label = singleton_groups[label]
        rng.shuffle(groups_for_label)
        for group, group_counts in groups_for_label:
            assign_group(group, group_counts)

    partitions = []
    for site_groups in assigned_groups:
        partition = [records[index] for group in site_groups for index in group]
        rng.shuffle(partition)
        partitions.append(partition)
    if any(not partition for partition in partitions):
        raise ValueError("Unable to give every site at least one leakage-independent training group")
    return partitions


def partition_records(records, num_sites=3, partition="iid", dirichlet_alpha=0.5, seed=42):
    """Partition records into deterministic, disjoint simulated sites."""

    records = _normalize_records(records)
    if num_sites < 2:
        raise ValueError("num_sites must be at least 2")
    if len(records) < num_sites:
        raise ValueError(f"Cannot partition {len(records)} records across {num_sites} sites")
    if partition == "iid":
        target_counts = _iid_target_counts(records, num_sites)
    elif partition == "dirichlet":
        target_counts = _dirichlet_target_counts(records, num_sites, dirichlet_alpha, seed)
    else:
        raise ValueError(f"Unknown partition strategy: {partition!r}")
    split_records = _group_aware_partition(records, num_sites, target_counts, seed)
    return {f"site-{index + 1}": site_records for index, site_records in enumerate(split_records)}


def _first_interval_overlap(left_records, right_records):
    left_intervals = sorted((*parse_genomic_interval(record["name"]), record["name"]) for record in left_records)
    right_intervals = sorted((*parse_genomic_interval(record["name"]), record["name"]) for record in right_records)
    left_index = 0
    right_index = 0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_chromosome, left_start, left_end, left_name = left_intervals[left_index]
        right_chromosome, right_start, right_end, right_name = right_intervals[right_index]
        if left_chromosome < right_chromosome:
            left_index += 1
        elif right_chromosome < left_chromosome:
            right_index += 1
        elif left_end <= right_start:
            left_index += 1
        elif right_end <= left_start:
            right_index += 1
        else:
            return left_name, right_name
    return None


def audit_split_leakage(splits):
    """Fail if exact sequences or genomic windows leak across named splits."""

    normalized_splits = {name: _normalize_records(records) for name, records in splits.items()}
    pair_reports = {}
    for left_name, right_name in combinations(normalized_splits, 2):
        left_records = normalized_splits[left_name]
        right_records = normalized_splits[right_name]

        left_sequences = {}
        for record in left_records:
            left_sequences.setdefault(_canonical_sequence(record), record["name"])
        for record in right_records:
            sequence = _canonical_sequence(record)
            if sequence in left_sequences:
                raise DataLeakageError(
                    f"Exact sequence leakage between {left_name!r} record {left_sequences[sequence]!r} "
                    f"and {right_name!r} record {record['name']!r}"
                )

        overlap = _first_interval_overlap(left_records, right_records)
        if overlap:
            raise DataLeakageError(
                f"Genomic interval leakage between {left_name!r} record {overlap[0]!r} "
                f"and {right_name!r} record {overlap[1]!r}"
            )
        pair_reports[f"{left_name}__{right_name}"] = {
            "exact_duplicate_sequences": 0,
            "genomic_interval_overlaps": 0,
        }
    return {"status": "passed", "split_pairs": pair_reports}


def prepare_records(
    train_records,
    test_records,
    num_sites=3,
    partition="iid",
    dirichlet_alpha=0.5,
    validation_fraction=0.1,
    showcase_size=3000,
    seed=42,
):
    """Prepare in-memory source records for writing or unit testing."""

    source_train = _normalize_records(train_records)
    source_test = _normalize_records(test_records)
    full_train, validation_records = stratified_train_validation_split(
        source_train, validation_fraction=validation_fraction, seed=seed
    )
    split_audit = audit_split_leakage({"train": full_train, "validation": validation_records, "test": source_test})
    selected_train = stratified_sample(full_train, showcase_size, seed=seed)
    sites = partition_records(
        selected_train,
        num_sites=num_sites,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    site_audit = audit_split_leakage(sites)
    return {
        "source_train_count": len(source_train),
        "available_train_count": len(full_train),
        "train": selected_train,
        "validation": validation_records,
        "test": source_test,
        "sites": sites,
        "audit": {
            "status": "passed",
            "split_pairs": split_audit["split_pairs"],
            "site_pairs": site_audit["split_pairs"],
        },
    }


def load_source_records(cache_dir=None):
    """Download only the pinned splice-site Parquet files from Hugging Face."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Data preparation requires the optional 'datasets' package") from exc

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        data_files=SOURCE_FILES,
        cache_dir=cache_dir,
    )
    missing_splits = sorted(set(SOURCE_FILES) - set(dataset))
    if missing_splits:
        raise ValueError(f"Downloaded dataset is missing splits: {', '.join(missing_splits)}")

    output = {}
    for split_name in SOURCE_FILES:
        task_records = [record for record in dataset[split_name] if record.get("task") == TASK]
        if not task_records:
            raise ValueError(f"Dataset split {split_name!r} contains no records for task {TASK!r}")
        output[split_name] = _normalize_records(task_records)
    _validate_source_contract(output)
    return output["train"], output["test"]


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps({field: record[field] for field in OUTPUT_FIELDS}, separators=(",", ":")))
            file.write("\n")


def _jsonl_file_identity(path):
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as file:
        for line in file:
            digest.update(line)
            rows += 1
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size, "rows": rows}


def write_prepared_data(
    output_dir,
    prepared,
    num_sites,
    partition,
    dirichlet_alpha,
    validation_fraction,
    showcase_size,
    seed,
):
    """Write prepared JSONL files and a reproducibility manifest."""

    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Remove generated files that belonged to an older preparation layout or
    # to sites omitted by this run. Keeping them would make the directory look
    # as though it still contained pooled or additional-site training data.
    expected_site_files = {f"{site_name}.jsonl" for site_name in prepared["sites"]}
    stale_train_files = [train_dir / "pooled.jsonl"]
    stale_train_files.extend(path for path in train_dir.glob("site-*.jsonl") if path.name not in expected_site_files)
    for stale_path in stale_train_files:
        if stale_path.is_file() or stale_path.is_symlink():
            stale_path.unlink()

    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    _write_jsonl(validation_path, prepared["validation"])
    _write_jsonl(test_path, prepared["test"])

    site_files = {}
    site_counts = {}
    site_identities = {}
    for site_name, site_records in prepared["sites"].items():
        site_path = train_dir / f"{site_name}.jsonl"
        _write_jsonl(site_path, site_records)
        site_files[site_name] = site_path.relative_to(output_dir).as_posix()
        site_identities[site_name] = _jsonl_file_identity(site_path)
        site_counts[site_name] = {
            "count": len(site_records),
            "label_histogram": _label_histogram(site_records),
        }

    manifest = {
        "format_version": 2,
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "task": TASK,
            "data_files": SOURCE_FILES,
        },
        "settings": {
            "num_sites": num_sites,
            "partition": partition,
            "dirichlet_alpha": dirichlet_alpha if partition == "dirichlet" else None,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "showcase_size": showcase_size,
        },
        "counts": {
            "source_train": prepared["source_train_count"],
            "available_train": prepared["available_train_count"],
            "train": len(prepared["train"]),
            "validation": len(prepared["validation"]),
            "test": len(prepared["test"]),
            "sites": site_counts,
        },
        "label_histograms": {
            "train": _label_histogram(prepared["train"]),
            "validation": _label_histogram(prepared["validation"]),
            "test": _label_histogram(prepared["test"]),
        },
        "audit": prepared["audit"],
        "files": {
            "validation": validation_path.relative_to(output_dir).as_posix(),
            "test": test_path.relative_to(output_dir).as_posix(),
            "sites": site_files,
        },
        "file_identities": {
            "validation": _jsonl_file_identity(validation_path),
            "test": _jsonl_file_identity(test_path),
            "sites": site_identities,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return manifest


def prepare_data(
    output_dir,
    num_sites=3,
    partition="iid",
    dirichlet_alpha=0.5,
    validation_fraction=0.1,
    showcase_size=3000,
    seed=42,
    cache_dir=None,
):
    """Download, validate, partition, and write the Evo2 splice-site dataset."""

    train_records, test_records = load_source_records(cache_dir=cache_dir)
    selected_showcase_size = showcase_size if showcase_size > 0 else None
    prepared = prepare_records(
        train_records,
        test_records,
        num_sites=num_sites,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        validation_fraction=validation_fraction,
        showcase_size=selected_showcase_size,
        seed=seed,
    )
    return write_prepared_data(
        output_dir,
        prepared,
        num_sites=num_sites,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        validation_fraction=validation_fraction,
        showcase_size=selected_showcase_size,
        seed=seed,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL data and manifest.json")
    parser.add_argument("--cache-dir", help="Optional Hugging Face datasets cache directory")
    parser.add_argument("--num-sites", type=int, default=3, help="Number of simulated institutions (default: 3)")
    parser.add_argument(
        "--partition",
        choices=("iid", "dirichlet"),
        default="iid",
        help="Training-data partition strategy (default: iid)",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration for label-skew partitions (default: 0.5)",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    parser.add_argument(
        "--showcase-size",
        type=int,
        default=3000,
        help="Stratified training subset size; use 0 for the full training split (default: 3000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = _parse_args()
    manifest = prepare_data(
        output_dir=args.output_dir,
        num_sites=args.num_sites,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        validation_fraction=args.validation_fraction,
        showcase_size=args.showcase_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
