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

"""Load fixed-schema TCIA features and normalize from matched training rows."""

import csv
import os
from numbers import Integral
from pathlib import Path

import numpy as np
from utils.normalization_utils import apply_standardizer, fit_standardizer

ID_COLUMNS = (
    "case_submitter_id",
    "case_id",
    "sample_submitter_id",
    "sample_id",
    "patient_id",
    "PatientID",
    "SubjectID",
)


def _identifier(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return "" if value is None else str(value).strip()


def _feature_mapping(ids, values):
    """Validate a rectangular numeric schema before associating rows with IDs."""
    ids = [_identifier(identifier) for identifier in ids]
    if any(not identifier for identifier in ids):
        raise ValueError("TCIA MRI feature rows must have nonempty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("TCIA MRI feature row IDs must be unique")
    if np.iscomplexobj(values):
        raise ValueError("TCIA MRI features must be real numbers")
    try:
        features = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("TCIA MRI features must form a rectangular numeric matrix") from exc
    if features.ndim != 2 or features.shape[0] != len(ids) or not all(features.shape):
        raise ValueError("TCIA MRI features require a nonempty 2D matrix with one row per ID")
    if not np.isfinite(features).all():
        raise ValueError("TCIA MRI features must not contain missing or non-finite values")
    return {identifier: features[index].copy() for index, identifier in enumerate(ids)}


def load_tcia_mri_features(path, id_column=None, normalize=False):
    """Read raw TCIA MRI features without fitting preprocessing to evaluation data.

    CSV/TSV requires one ID column and fixed, numeric feature columns. NPZ
    requires a 2D ``features`` array and 1D ``case_ids``, ``sample_ids`` or
    ``ids``. Normalize with ``normalize_training_mri_features`` after splitting
    patients; table-wide normalization is intentionally unsupported.
    """
    if normalize:
        raise ValueError(
            "Use normalize_training_mri_features with training indices; whole-table normalization is disabled"
        )
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"TCIA MRI feature file not found: {path}")

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "features" not in data:
                raise ValueError("NPZ TCIA MRI features must include features")
            id_key = next((key for key in ("case_ids", "sample_ids", "ids") if key in data), None)
            if id_key is None:
                raise ValueError("NPZ TCIA MRI features must include case_ids, sample_ids, or ids")
            ids = data[id_key]
            if ids.ndim != 1:
                raise ValueError("NPZ TCIA MRI IDs must be a 1D array")
            return _feature_mapping(ids.tolist(), data["features"])

    delimiter = "\t" if suffix == ".tsv" else ","
    ids, values = [], []
    with open(path, newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)) or any(not name for name in fieldnames):
            raise ValueError("TCIA MRI table columns must have unique nonempty names")
        selected_id = id_column or next((column for column in ID_COLUMNS if column in fieldnames), None)
        if selected_id not in fieldnames:
            raise ValueError("TCIA MRI feature table needs an ID column; provide id_column or a supported ID header")
        feature_columns = [column for column in fieldnames if column != selected_id]
        if not feature_columns:
            raise ValueError("TCIA MRI table requires at least one numeric feature column")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"TCIA MRI row {row_number} has more values than the header schema")
            ids.append(row[selected_id])
            feature_row = []
            for column in feature_columns:
                raw = row.get(column)
                if raw is None or not raw.strip():
                    raise ValueError(f"TCIA MRI row {row_number} is missing feature {column!r}")
                try:
                    feature_row.append(float(raw))
                except ValueError as exc:
                    raise ValueError(f"TCIA MRI row {row_number} has nonnumeric feature {column!r}") from exc
            values.append(feature_row)
    return _feature_mapping(ids, values)


def _match_feature_id(record, feature_by_id):
    """Use the same case/sample/file matching order for fitting and pooling."""
    raw_candidates = (
        record.get("case_id"),
        record.get("case_submitter_id"),
        record.get("sample_id"),
        record.get("sample_submitter_id"),
        record.get("file_id"),
    )
    seen = set()
    for value in raw_candidates:
        identifier = _identifier(value)
        candidates = [identifier]
        if identifier.startswith("TCGA-"):
            parts = identifier.split("-")
            if len(parts) >= 3:
                candidates.append("-".join(parts[:3]))
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                if candidate in feature_by_id:
                    return candidate
    return None


def normalize_training_mri_features(records, train_indices, feature_by_id):
    """Fit on unique TCIA rows matched by training records, then transform all rows.

    Paired Normal/Tumor expression samples and repeated expression files do not
    increase a TCIA patient's weight in the fitted statistics. Unmatched rows,
    including evaluation-only patients, never contribute to the fit.
    """
    if not feature_by_id:
        raise ValueError("TCIA MRI normalization requires a nonempty feature table")
    features = _feature_mapping(list(feature_by_id), list(feature_by_id.values()))
    fit_ids = set()
    for index in train_indices:
        if not isinstance(index, Integral) or not 0 <= index < len(records):
            raise ValueError(f"Invalid training record index: {index}")
        matched = _match_feature_id(records[index], features)
        if matched is not None:
            fit_ids.add(matched)
    if not fit_ids:
        raise ValueError("No TCIA MRI feature rows match the training records; cannot fit normalization")
    fit_ids = sorted(fit_ids)
    stats = fit_standardizer(np.stack([features[identifier] for identifier in fit_ids]))
    all_ids = list(features)
    normalized = apply_standardizer(np.stack([features[identifier] for identifier in all_ids]), stats)
    audit = {**stats, "fit_ids": fit_ids}
    return {identifier: normalized[index].copy() for index, identifier in enumerate(all_ids)}, audit


def aggregate_mri_features(records, indices, feature_by_id, feature_dim=None):
    """Mean-pool MRI features for the samples assigned to one client/task."""
    vectors = []
    matched_ids = []
    for index in indices:
        identifier = _match_feature_id(records[index], feature_by_id)
        if identifier is not None:
            vectors.append(feature_by_id[identifier])
            matched_ids.append(identifier)

    if vectors:
        matrix = np.vstack(vectors).astype(np.float32)
        return matrix.mean(axis=0), matched_ids

    if feature_dim is None:
        first = next(iter(feature_by_id.values()))
        feature_dim = int(first.shape[0])
    return np.zeros(feature_dim, dtype=np.float32), matched_ids
