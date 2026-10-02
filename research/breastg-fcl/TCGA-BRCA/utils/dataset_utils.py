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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import hashlib
import json
import os
import pickle
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from utils.normalization_utils import apply_standardizer, fit_standardizer
from utils.tcia_mri_utils import aggregate_mri_features, load_tcia_mri_features, normalize_training_mri_features

LABEL_MAP = {
    "Normal": 0,
    "Tumor": 1,
}


CLINICAL_STAGE_TASKS = [
    {
        "id": 0,
        "name": "early_stage",
        "description": "AJCC Stage 0/I breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage 0", "stage i"),
    },
    {
        "id": 1,
        "name": "intermediate_stage",
        "description": "AJCC Stage II breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage ii",),
    },
    {
        "id": 2,
        "name": "advanced_stage",
        "description": "AJCC Stage III/IV or metastatic breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage iii", "stage iv"),
    },
]


def _load_clinical_cases(path):
    clinical = {}
    if not path or not os.path.exists(path):
        return clinical
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_submitter_id = row.get("case_submitter_id", "")
            if case_submitter_id:
                clinical[case_submitter_id] = row
    return clinical


def _normalize_stage(stage):
    return " ".join(str(stage or "").strip().lower().split())


def _clinical_stage_task_id(stage, sample_type=""):
    sample_type = str(sample_type or "")
    normalized = _normalize_stage(stage)
    if sample_type == "Metastatic" or normalized.startswith("stage iv"):
        return 2
    if normalized.startswith("stage iii"):
        return 2
    if normalized.startswith("stage ii"):
        return 1
    if normalized.startswith("stage 0") or normalized.startswith("stage i"):
        return 0
    return None


def _split_index_list(indices, train_split, rng):
    shuffled = list(indices)
    rng.shuffle(shuffled)
    if len(shuffled) > 1:
        split = int(round(len(shuffled) * train_split))
        split = min(max(split, 1), len(shuffled) - 1)
    else:
        split = len(shuffled)
    train = shuffled[:split]
    test = shuffled[split:]
    return train, test


def _record_identities(records):
    """Require patient/sample identities rather than treating files as patients."""
    patient_ids, sample_ids = [], []
    sample_patients = {}
    for index, record in enumerate(records):
        patient = str(record.get("case_submitter_id") or "").strip().upper()
        sample = str(record.get("sample_submitter_id") or record.get("sample_id") or "").strip().upper()
        if not patient or not sample:
            raise ValueError(f"Record {index} requires case_submitter_id and sample_submitter_id for patient splitting")
        if sample.startswith("TCGA-"):
            parts = sample.split("-")
            if len(parts) < 4 or "-".join(parts[:3]) != patient:
                raise ValueError(f"Sample {sample} does not belong to patient {patient}")
        if sample in sample_patients and sample_patients[sample] != patient:
            raise ValueError(f"Sample {sample} is associated with multiple patients")
        sample_patients[sample] = patient
        patient_ids.append(patient)
        sample_ids.append(sample)
    return patient_ids, sample_ids


def _assert_disjoint_patient_sample_ids(records, train_indices, test_indices):
    """Validate global holdout isolation, including all tasks and all clients."""
    patient_ids, sample_ids = _record_identities(records)
    identities = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        if any(not isinstance(index, (int, np.integer)) or not 0 <= index < len(records) for index in indices):
            raise ValueError(f"Invalid record index in {split} partition")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate record indices in {split} partition")
        identities[f"{split}_patient_ids"] = sorted({patient_ids[index] for index in indices})
        identities[f"{split}_sample_ids"] = sorted({sample_ids[index] for index in indices})
    for unit in ("patient", "sample"):
        overlap = set(identities[f"train_{unit}_ids"]) & set(identities[f"test_{unit}_ids"])
        if overlap:
            raise ValueError(f"Train/test {unit} identities overlap: {sorted(overlap)[:5]}")
    return identities


def _split_patient_indices(records, targets, train_split, seed):
    """Split patients globally, stratifying by each patient's set of labels.

    All files for a patient, including paired Normal/Tumor samples, stay on
    one side. A singleton stratum stays in training and is never copied to
    evaluation. Task and client assignment happen only after this split.
    """
    if not 0 < train_split < 1:
        raise ValueError("train_split must be strictly between 0 and 1")
    if len(records) != len(targets):
        raise ValueError("Records and targets must have the same length")
    patient_ids, _ = _record_identities(records)
    patient_labels = defaultdict(set)
    for patient, label in zip(patient_ids, targets):
        patient_labels[patient].add(int(label))
    strata = defaultdict(list)
    for patient in sorted(patient_labels):
        strata[tuple(sorted(patient_labels[patient]))].append(patient)
    rng = np.random.default_rng(seed)
    train_patients, test_patients = set(), set()
    for labels in sorted(strata):
        train, test = _split_index_list(strata[labels], train_split, rng)
        train_patients.update(train)
        test_patients.update(test)
    if not train_patients or not test_patients:
        raise ValueError("Patient split requires nonempty train/test groups; provide more independent patients")
    train = [index for index, patient in enumerate(patient_ids) if patient in train_patients]
    test = [index for index, patient in enumerate(patient_ids) if patient in test_patients]
    _assert_disjoint_patient_sample_ids(records, train, test)
    return train, test


def _validate_client_task_split(records, train_indices, test_indices, opt):
    """Reject incomplete cells and any identity leakage after distribution."""
    flattened = {}
    for split, assignments in (("train", train_indices), ("test", test_indices)):
        flattened[split] = []
        for client_id in range(opt.num_clients):
            for task_id in range(opt.num_task):
                indices = assignments.get(client_id, {}).get(task_id, [])
                if not indices:
                    raise ValueError(
                        f"Empty {split} partition for client {client_id}, task {task_id} after patient splitting; "
                        "provide more patients or reduce the client/task count. Samples will not be duplicated "
                        "or borrowed from another task or split."
                    )
                flattened[split].extend(indices)
    return _assert_disjoint_patient_sample_ids(records, flattened["train"], flattened["test"])


def _distribute_task_label_indices(task_label_indices, opt, seed):
    rng = np.random.default_rng(seed)
    client_task_indices = defaultdict(dict)
    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            client_task_indices[client_id][task_id] = []

    for task_id in range(opt.num_task):
        offset = 0
        for indices in task_label_indices.get(task_id, {}).values():
            label_buckets = _assign_to_buckets(indices, opt.num_clients, rng)
            for bucket_id, label_indices in enumerate(label_buckets):
                client_id = (offset + bucket_id) % opt.num_clients
                client_task_indices[client_id][task_id].extend(label_indices)
            offset += len(indices)

        for client_id in range(opt.num_clients):
            bucket = client_task_indices[client_id][task_id]
            rng.shuffle(bucket)

    return client_task_indices


def _task_label_summary(indices_by_task_label):
    summary = {}
    for task_id, by_label in indices_by_task_label.items():
        summary[str(int(task_id))] = {
            label_name: int(len(by_label.get(label_id, []))) for label_name, label_id in LABEL_MAP.items()
        }
    return summary


def _make_clinical_stage_task_indices(records, targets, opt):
    if opt.num_task != len(CLINICAL_STAGE_TASKS):
        raise ValueError(
            "clinical_stage task split expects --num-task 3: " "early_stage, intermediate_stage, advanced_stage"
        )

    train_records, test_records = _split_patient_indices(records, targets, opt.train_split, opt.seed)
    clinical = _load_clinical_cases(getattr(opt, "clinical_path", None))
    if not clinical:
        raise RuntimeError(
            "clinical_stage task split requires GDC clinical metadata. "
            "Provide --clinical-path pointing to gdc_clinical_cases.tsv."
        )

    normal_indices = []
    tumor_by_task = {task["id"]: [] for task in CLINICAL_STAGE_TASKS}
    excluded_unknown_stage = []

    for idx, record in enumerate(records):
        label = int(targets[idx])
        if label == LABEL_MAP["Normal"]:
            normal_indices.append(idx)
            continue

        case_submitter_id = record.get("case_submitter_id", "")
        clinical_row = clinical.get(case_submitter_id, {})
        stage = clinical_row.get("ajcc_pathologic_stage", "")
        task_id = _clinical_stage_task_id(stage, record.get("sample_type", ""))
        if task_id is None:
            if getattr(opt, "include_unknown_stage", False):
                task_id = len(CLINICAL_STAGE_TASKS) - 1
            else:
                excluded_unknown_stage.append(idx)
                continue
        tumor_by_task[task_id].append(idx)
        record["ajcc_pathologic_stage"] = stage
        record["clinical_task_id"] = task_id
        record["clinical_task_name"] = CLINICAL_STAGE_TASKS[task_id]["name"]

    train_by_task_label = defaultdict(lambda: defaultdict(list))
    test_by_task_label = defaultdict(lambda: defaultdict(list))

    for members, by_task_label, seed in (
        (set(train_records), train_by_task_label, opt.seed + 2),
        (set(test_records), test_by_task_label, opt.seed + 3),
    ):
        normal_buckets = _assign_to_buckets(
            [index for index in normal_indices if index in members], opt.num_task, np.random.default_rng(seed)
        )
        for task in CLINICAL_STAGE_TASKS:
            task_id = task["id"]
            by_task_label[task_id][LABEL_MAP["Normal"]] = normal_buckets[task_id]
            by_task_label[task_id][LABEL_MAP["Tumor"]] = [index for index in tumor_by_task[task_id] if index in members]

    train_indices = _distribute_task_label_indices(train_by_task_label, opt, opt.seed)
    test_indices = _distribute_task_label_indices(test_by_task_label, opt, opt.seed + 1)

    opt.task_metadata = [
        {
            "task_id": task["id"],
            "task_name": task["name"],
            "description": task["description"],
            "stage_prefixes": list(task["stage_prefixes"]),
        }
        for task in CLINICAL_STAGE_TASKS
    ]
    opt.task_label_counts = _task_label_summary(train_by_task_label)
    opt.task_test_label_counts = _task_label_summary(test_by_task_label)
    opt.excluded_unknown_stage_count = int(len(excluded_unknown_stage))
    _validate_client_task_split(records, train_indices, test_indices, opt)
    return train_indices, test_indices


def _make_random_task_indices(records, targets, opt):
    train_records, test_records = _split_patient_indices(records, targets, opt.train_split, opt.seed)
    train_by_label, test_by_label = defaultdict(list), defaultdict(list)
    for index in train_records:
        train_by_label[int(targets[index])].append(index)
    for index in test_records:
        test_by_label[int(targets[index])].append(index)
    train_indices = _make_client_task_indices(train_by_label, opt, opt.seed)
    test_indices = _make_client_task_indices(test_by_label, opt, opt.seed + 1)
    opt.task_metadata = [
        {
            "task_id": task_id,
            "task_name": f"random_task_{task_id}",
            "description": "Random TCGA-BRCA tasks within a patient-disjoint train/test split",
        }
        for task_id in range(opt.num_task)
    ]
    _validate_client_task_split(records, train_indices, test_indices, opt)
    return train_indices, test_indices


class TCGABRCADataset(Dataset):
    """TCGA-BRCA RNA-seq expression dataset backed by an in-memory matrix."""

    def __init__(self, features, targets, sample_ids=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.sample_ids = sample_ids or [str(i) for i in range(len(self.targets))]

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def _read_manifest(manifest_path, raw_dir):
    records = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tissue_type = row.get("tissue_type", "")
            if tissue_type not in LABEL_MAP:
                continue

            file_id = row["file_id"]
            file_name = row["file_name"]
            file_path = os.path.join(raw_dir, file_id, file_name)
            if not os.path.exists(file_path):
                continue

            records.append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "sample_id": row.get("sample_submitter_id", ""),
                    "sample_submitter_id": row.get("sample_submitter_id", ""),
                    "case_id": row.get("case_submitter_id", ""),
                    "case_submitter_id": row.get("case_submitter_id", ""),
                    "sample_type": row.get("sample_type", ""),
                    "label_name": tissue_type,
                    "label": LABEL_MAP[tissue_type],
                }
            )
    if not records:
        raise RuntimeError(f"No usable TCGA-BRCA expression files found from manifest: {manifest_path}")
    return records


def _read_expression_vector(path, value_column, max_genes):
    values = []
    gene_ids = []
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#")),
            delimiter="\t",
        )
        if value_column not in reader.fieldnames:
            raise ValueError(f"{value_column} not found in {path}")

        for row in reader:
            if row.get("gene_type") != "protein_coding":
                continue
            raw_value = row.get(value_column, "")
            value = 0.0 if raw_value == "" else float(raw_value)
            gene_ids.append(row.get("gene_id", ""))
            values.append(value)
            if len(values) >= max_genes:
                break

    if len(values) < max_genes:
        values.extend([0.0] * (max_genes - len(values)))
        gene_ids.extend([""] * (max_genes - len(gene_ids)))

    return np.asarray(values, dtype=np.float32), gene_ids


def _expression_source_sha256(path):
    """Hash actual source bytes, including on cache hits; do not trust timestamps."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expression_cache_metadata(opt, records):
    """Bind every cached row to its ordered identities, label, and source bytes."""
    if not records or opt.max_genes < 1:
        raise ValueError("Expression cache requires records and a positive gene count")
    patient_ids, sample_ids = _record_identities(records)
    metadata = {
        "cache_schema_version": np.asarray(3),
        "preprocessing": np.asarray("log1p_only"),
        "expression_value_col": np.asarray(opt.expression_value_col),
        "max_genes": np.asarray(opt.max_genes),
        "file_ids": np.asarray([str(record["file_id"]) for record in records]),
        "patient_ids": np.asarray(patient_ids),
        "sample_submitter_ids": np.asarray(sample_ids),
        "sample_ids": np.asarray([str(record["sample_id"]) for record in records]),
        "targets": np.asarray([record["label"] for record in records], dtype=np.int64),
        "source_sha256": np.asarray([_expression_source_sha256(record["file_path"]) for record in records]),
    }
    serialized = json.dumps(
        {key: value.tolist() for key, value in metadata.items()}, sort_keys=True, separators=(",", ":")
    )
    return metadata, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_expression_cache(cache_path, metadata, fingerprint):
    """Reject incomplete or mismatched caches before returning any expression rows."""
    try:
        data = np.load(cache_path, allow_pickle=False)
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError("expected an NPZ archive")
        with data:
            if data["record_fingerprint"].shape != () or data["record_fingerprint"].item() != fingerprint:
                raise ValueError("record fingerprint does not match the ordered inputs")
            for key, expected in metadata.items():
                actual = data[key]
                if actual.dtype.kind != expected.dtype.kind or not np.array_equal(actual, expected):
                    raise ValueError(f"{key} does not match the ordered inputs")
            features, gene_ids = data["features"], data["gene_ids"]
            shape = (len(metadata["sample_ids"]), int(metadata["max_genes"]))
            if features.dtype != np.float32 or features.shape != shape or not np.isfinite(features).all():
                raise ValueError(f"features must be a finite float32 matrix of shape {shape}")
            if gene_ids.dtype.kind != "U" or gene_ids.shape != (shape[1],):
                raise ValueError("gene identities do not match the feature columns")
            return features, data["targets"].astype(np.int64), data["sample_ids"].tolist(), gene_ids.tolist()
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid expression cache {cache_path}: {exc}. Remove it and rebuild from source.") from exc


def _build_expression_cache(opt, records):
    """Cache sample-local log1p values with verified input identities and checksums."""
    cache_dir = os.path.join(opt.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    metadata, fingerprint = _expression_cache_metadata(opt, records)
    cache_name = (
        f"tcga_brca_log1p_v3_{opt.expression_value_col}_{opt.max_genes}_"
        f"{len(records)}samples_{fingerprint[:16]}.npz"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        return _load_expression_cache(cache_path, metadata, fingerprint)

    feature_rows = []
    gene_ids = None

    for index, record in enumerate(records):
        vector, current_gene_ids = _read_expression_vector(
            record["file_path"],
            opt.expression_value_col,
            opt.max_genes,
        )
        if _expression_source_sha256(record["file_path"]) != metadata["source_sha256"][index]:
            raise ValueError(f"Expression source changed while building the cache: {record['file_path']}")
        if gene_ids is None:
            gene_ids = current_gene_ids
        elif current_gene_ids != gene_ids:
            raise ValueError(f"Expression gene identities/order differ in {record['file_path']}")
        if vector.shape != (opt.max_genes,) or len(current_gene_ids) != opt.max_genes:
            raise ValueError(f"Expression source has an unexpected gene count: {record['file_path']}")
        feature_rows.append(np.log1p(vector))

    features = np.vstack(feature_rows).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Expression sources produced non-finite log1p values")
    # Only publish a complete archive, so interrupted/concurrent writers cannot
    # expose a partially written cache to another loader.
    descriptor, temporary_path = tempfile.mkstemp(prefix=".tcga_brca_", suffix=".npz", dir=cache_dir)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                features=features,
                gene_ids=np.asarray(gene_ids, dtype=str),
                record_fingerprint=fingerprint,
                **metadata,
            )
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return features, metadata["targets"], metadata["sample_ids"].tolist(), gene_ids


def _assign_to_buckets(indices, num_buckets, rng):
    if num_buckets < 1:
        raise ValueError("The number of client/task buckets must be positive")
    shuffled = list(indices)
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(num_buckets)]
    for i, idx in enumerate(shuffled):
        buckets[i % num_buckets].append(idx)
    return buckets


def _make_client_task_indices(indices_by_label, opt, seed):
    rng = np.random.default_rng(seed)
    num_buckets = opt.num_clients * opt.num_task
    buckets = [[] for _ in range(num_buckets)]

    offset = 0
    for indices in indices_by_label.values():
        label_buckets = _assign_to_buckets(indices, num_buckets, rng)
        for bucket_id, label_indices in enumerate(label_buckets):
            buckets[(offset + bucket_id) % num_buckets].extend(label_indices)
        offset += len(indices)

    client_task_indices = defaultdict(dict)
    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            bucket_id = client_id * opt.num_task + task_id
            rng.shuffle(buckets[bucket_id])
            client_task_indices[client_id][task_id] = buckets[bucket_id]

    return client_task_indices


def setup_tcga_brca_loaders(opt):
    """Create TCGA-BRCA federated loaders from GDC STAR gene-count files."""
    os.makedirs(opt.output_dir, exist_ok=True)

    manifest_path = getattr(
        opt,
        "manifest_path",
        os.path.join(opt.data_dir, "..", "metadata", "gdc_files_manifest.tsv"),
    )
    raw_dir = getattr(opt, "raw_dir", os.path.join(opt.data_dir, "raw"))

    records = _read_manifest(manifest_path, raw_dir)
    targets = np.asarray([record["label"] for record in records], dtype=np.int64)
    opt.num_classes = int(len(set(targets.tolist())))
    opt.nc = opt.num_classes

    if getattr(opt, "task_split_strategy", "clinical_stage") == "clinical_stage":
        train_indices, test_indices = _make_clinical_stage_task_indices(records, targets, opt)
    else:
        train_indices, test_indices = _make_random_task_indices(records, targets, opt)
    identity_split = _validate_client_task_split(records, train_indices, test_indices, opt)
    features, cached_targets, sample_ids, gene_ids = _build_expression_cache(opt, records)
    if not np.array_equal(cached_targets, targets) or sample_ids != [record["sample_id"] for record in records]:
        raise ValueError("Expression cache labels/sample identities do not match the current manifest")
    # Only records used for training may contribute to fitted statistics.
    # Excluded-stage tumors and evaluation patients never enter this matrix.
    training_rows = sorted(index for tasks in train_indices.values() for indices in tasks.values() for index in indices)
    expression_statistics = fit_standardizer(features[training_rows])
    features = apply_standardizer(features, expression_statistics)
    expression_statistics["fit_indices"] = training_rows
    opt.input_dim = int(features.shape[1])
    dataset = TCGABRCADataset(features, targets, sample_ids)

    feature_by_id = None
    opt.client_spatial_features = None
    opt.client_spatial_feature_matches = None
    spatial_path = getattr(opt, "tcia_mri_features_path", None)
    if not spatial_path or not os.path.exists(spatial_path):
        raise RuntimeError("BreastG-FCL requires TCIA spatial features; " "provide --tcia-mri-features-path")
    feature_by_id = load_tcia_mri_features(
        spatial_path,
        id_column=getattr(opt, "tcia_mri_id_column", None),
        normalize=False,
    )
    feature_by_id, spatial_statistics = normalize_training_mri_features(records, training_rows, feature_by_id)

    if feature_by_id:
        feature_dim = int(next(iter(feature_by_id.values())).shape[0])
        spatial_by_task = []
        match_counts_by_task = []
        for task_id in range(opt.num_task):
            task_vectors = []
            task_match_counts = []
            for client_id in range(opt.num_clients):
                vector, matched_ids = aggregate_mri_features(
                    records,
                    train_indices[client_id][task_id],
                    feature_by_id,
                    feature_dim=feature_dim,
                )
                task_vectors.append(vector)
                task_match_counts.append(len(matched_ids))
            spatial_by_task.append(np.vstack(task_vectors).astype(np.float32))
            match_counts_by_task.append(task_match_counts)
        opt.client_spatial_features = spatial_by_task
        opt.client_spatial_feature_matches = match_counts_by_task

    temporal_feature_by_id = None
    opt.client_temporal_features = None
    opt.client_temporal_feature_matches = None
    kinetics_path = getattr(opt, "tcia_dce_kinetics_path", None)
    if not kinetics_path or not os.path.exists(kinetics_path):
        raise RuntimeError("BreastG-FCL requires DCE temporal features; " "provide --tcia-dce-kinetics-path")
    temporal_feature_by_id = load_tcia_mri_features(
        kinetics_path,
        id_column=getattr(opt, "tcia_dce_id_column", None),
        normalize=False,
    )
    temporal_feature_by_id, temporal_statistics = normalize_training_mri_features(
        records, training_rows, temporal_feature_by_id
    )

    if temporal_feature_by_id:
        temporal_dim = int(next(iter(temporal_feature_by_id.values())).shape[0])
        temporal_by_task = []
        temporal_match_counts_by_task = []
        for task_id in range(opt.num_task):
            task_vectors = []
            task_match_counts = []
            for client_id in range(opt.num_clients):
                vector, matched_ids = aggregate_mri_features(
                    records,
                    train_indices[client_id][task_id],
                    temporal_feature_by_id,
                    feature_dim=temporal_dim,
                )
                task_vectors.append(vector)
                task_match_counts.append(len(matched_ids))
            temporal_by_task.append(np.vstack(task_vectors).astype(np.float32))
            temporal_match_counts_by_task.append(task_match_counts)
        opt.client_temporal_features = temporal_by_task
        opt.client_temporal_feature_matches = temporal_match_counts_by_task

    client_loaders = defaultdict(dict)
    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            client_loaders[client_id][task_id] = {
                "train": DataLoader(
                    Subset(dataset, train_indices[client_id][task_id]),
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    num_workers=opt.num_workers,
                    pin_memory=opt.pin_memory,
                ),
                "test": DataLoader(
                    Subset(dataset, test_indices[client_id][task_id]),
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    pin_memory=opt.pin_memory,
                ),
            }

    label_counts = {label_name: int((targets == label_id).sum()) for label_name, label_id in LABEL_MAP.items()}
    patient_ids, identity_sample_ids = _record_identities(records)
    partitioning = {
        "dataset": "TCGA-BRCA",
        "label_map": LABEL_MAP,
        "label_counts": label_counts,
        "num_clients": opt.num_clients,
        "tasks_per_client": opt.num_task,
        "task_split_strategy": getattr(opt, "task_split_strategy", None),
        "split_unit": "patient",
        "split_protocol_version": 2,
        "identity_split": identity_split,
        "normalization": {
            "schema_version": 1,
            "expression": expression_statistics,
            "tcia_spatial": spatial_statistics,
            "tcia_temporal": temporal_statistics,
        },
        "patient_ids_by_index": patient_ids,
        "sample_ids_by_index": identity_sample_ids,
        "file_ids_by_index": [record["file_id"] for record in records],
        "task_metadata": getattr(opt, "task_metadata", None),
        "task_label_counts": getattr(opt, "task_label_counts", None),
        "task_test_label_counts": getattr(opt, "task_test_label_counts", None),
        "excluded_unknown_stage_count": getattr(opt, "excluded_unknown_stage_count", 0),
        "train_split": opt.train_split,
        "input_dim": opt.input_dim,
        "expression_value_col": opt.expression_value_col,
        "max_genes": opt.max_genes,
        "gene_ids": gene_ids,
        "spatial_attention_source": getattr(opt, "spatial_attention_source", None),
        "tcia_mri_features_path": getattr(opt, "tcia_mri_features_path", None),
        "client_spatial_feature_matches": getattr(opt, "client_spatial_feature_matches", None),
        "temporal_attention_source": getattr(opt, "temporal_attention_source", None),
        "tcia_dce_kinetics_path": getattr(opt, "tcia_dce_kinetics_path", None),
        "client_temporal_feature_matches": getattr(opt, "client_temporal_feature_matches", None),
        "client_task_train_indices": dict(train_indices),
        "client_task_test_indices": dict(test_indices),
        "seed": opt.seed,
    }

    partition_path = os.path.join(
        opt.output_dir,
        f"tcga_brca_partitioning_seed{opt.seed}.pkl",
    )
    with open(partition_path, "wb") as f:
        pickle.dump(partitioning, f)

    print("Created TCGA-BRCA dataloaders:")
    print(f"  - samples: {len(dataset)}")
    print(f"  - labels: {label_counts}")
    print(f"  - clients: {opt.num_clients}")
    print(f"  - tasks per client: {opt.num_task}")
    print(f"  - task split strategy: {getattr(opt, 'task_split_strategy', None)}")
    print(
        f"  - disjoint train/test patients: {len(identity_split['train_patient_ids'])}/"
        f"{len(identity_split['test_patient_ids'])} (sample identities also disjoint)"
    )
    if getattr(opt, "task_metadata", None):
        for task in opt.task_metadata:
            task_id = task["task_id"]
            task_key = str(task_id)
            train_counts = getattr(opt, "task_label_counts", {}).get(task_key, {})
            test_counts = getattr(opt, "task_test_label_counts", {}).get(task_key, {})
            print(f"  - task {task_id} {task['task_name']}: " f"train={train_counts}, test={test_counts}")
    if getattr(opt, "excluded_unknown_stage_count", 0):
        print(f"  - excluded unknown-stage tumor samples: {opt.excluded_unknown_stage_count}")
    print(f"  - input_dim: {opt.input_dim}")
    print(
        f"  - normalization fitted on training data only: {expression_statistics['n_samples_seen']} expression rows, "
        f"{spatial_statistics['n_samples_seen']} spatial rows, {temporal_statistics['n_samples_seen']} temporal rows"
    )
    if getattr(opt, "client_spatial_features", None) is not None:
        print(f"  - TCIA MRI spatial feature dim: {opt.client_spatial_features[0].shape[1]}")
    if getattr(opt, "client_temporal_features", None) is not None:
        print(f"  - DCE kinetic temporal feature dim: {opt.client_temporal_features[0].shape[1]}")

    return client_loaders
