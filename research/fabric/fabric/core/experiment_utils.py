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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Experiment descriptions, robust feature reads, and cross-validation summaries."""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from . import training
from .client_training import Condition
from .training import compute_metrics, metric_rows_from_predictions, summarize_outputs

METRICS = ("auc", "pr_auc", "bacc", "acc", "sensitivity", "specificity", "test_loss")


COUNT_COLS = ("predicted_positives", "true_positives_detected", "false_positives", "false_negatives", "n_test")


ORIGINAL_READ_FEATURE_ARRAY = training.read_feature_array


HDF5_LOG_PATH: Path | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    encoder: str
    mil_label: str
    model_name: str
    source_run_dir: Path
    out_dir: Path


MODEL_SPECS = {
    f"{key}_dtfd": ModelSpec(
        key=f"{key}_dtfd",
        display_name=f"{encoder} + DTFD-MIL FedAvg epoch5",
        encoder=encoder,
        mil_label="DTFD-MIL",
        model_name="dtfd_mil",
        source_run_dir=Path(),
        out_dir=Path(),
    )
    for key, encoder in (("uni", "UNI"), ("virchow2", "Virchow2"))
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_to_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, handle)
        sys.stderr = Tee(old_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def log_hdf5(message: str) -> None:
    print(message, flush=True)
    if HDF5_LOG_PATH is not None:
        HDF5_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HDF5_LOG_PATH.open("a", buffering=1) as handle:
            handle.write(f"{datetime.now().isoformat(timespec='seconds')}\t{message}\n")


def robust_hdf5_feature_read(path: Path, row_indices: np.ndarray | None = None) -> torch.Tensor:
    """Read HDF5 features while skipping only unreadable rows."""
    with h5py.File(path, "r") as handle:
        key = next((k for k in ["features", "feature", "feat", "x"] if k in handle), None)
        if key is None:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f"No datasets in {path}")
            key = keys[0]
        dataset = handle[key]

        if row_indices is not None:
            rows = []
            skipped = []
            for idx in np.asarray(row_indices, dtype=np.int64).tolist():
                try:
                    rows.append(dataset[int(idx)])
                except OSError:
                    skipped.append(int(idx))
            if skipped:
                log_hdf5(f"[hdf5-fallback] skipped {len(skipped)} selected rows in {path}: {skipped[:10]}")
            if rows:
                values = np.stack(rows, axis=0)
            else:
                feature_dim = int(dataset.shape[-1]) if len(dataset.shape) > 1 else 1
                values = np.empty((0, feature_dim), dtype=dataset.dtype)
            return training.normalize_feature_tensor(values, path)

        chunks = []
        chunk_size = 512
        for start in range(0, int(dataset.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(dataset.shape[0]))
            try:
                chunks.append(dataset[start:stop])
            except OSError:
                rows = []
                skipped = []
                for idx in range(start, stop):
                    try:
                        rows.append(dataset[idx])
                    except OSError:
                        skipped.append(idx)
                if skipped:
                    log_hdf5(
                        f"[hdf5-fallback] skipped {len(skipped)} unreadable rows in {path}: "
                        f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}"
                    )
                if rows:
                    chunks.append(np.stack(rows, axis=0))

        if chunks:
            values = np.concatenate(chunks, axis=0)
        else:
            feature_dim = int(dataset.shape[-1]) if len(dataset.shape) > 1 else 1
            values = np.empty((0, feature_dim), dtype=dataset.dtype)
        return training.normalize_feature_tensor(values, path)


def safe_read_feature_array(path: str | Path, row_indices: np.ndarray | None = None) -> torch.Tensor:
    path = Path(path)
    try:
        return ORIGINAL_READ_FEATURE_ARRAY(path, row_indices=row_indices)
    except OSError as exc:
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise
        log_hdf5(f"[hdf5-fallback] {path}: {exc}")
        return robust_hdf5_feature_read(path, row_indices=row_indices)


def install_hdf5_safe_reader(log_path: Path) -> None:
    global HDF5_LOG_PATH
    HDF5_LOG_PATH = log_path
    training.read_feature_array = safe_read_feature_array
    log_hdf5(f"[hdf5-fallback] safe reader installed; log={log_path}")


def condition_for(spec: ModelSpec) -> Condition:
    return Condition(
        key=spec.key,
        label=f"{spec.display_name} local_epochs=5 class-weighted CE normal sampling",
        algorithm="fedavg",
        dirname=spec.out_dir.name,
        mu=0.0,
        source_fold0_dir=Path(),
    )


def count_predictions(pred_df: pd.DataFrame, threshold: float) -> dict:
    labels = pred_df["recurrence_label"].astype(int).to_numpy()
    probs = pred_df["prob_recurrence"].astype(float).to_numpy()
    preds = (probs >= float(threshold)).astype(int)
    return {
        "predicted_positives": int(preds.sum()),
        "true_positives_detected": int(((labels == 1) & (preds == 1)).sum()),
        "true_positives_total": int((labels == 1).sum()),
        "false_positives": int(((labels == 0) & (preds == 1)).sum()),
        "false_negatives": int(((labels == 1) & (preds == 0)).sum()),
        "n_test": int(len(pred_df)),
    }


def normalized_result_row(
    spec: ModelSpec, row: dict, fold: int, pred_df: pd.DataFrame, threshold: float, out_dir: Path
) -> dict:
    row = dict(row)
    test_loss = float(row.get("test_loss", row.get("loss", float("nan"))))
    metrics = compute_metrics(
        pred_df["recurrence_label"], pred_df["prob_recurrence"], threshold=float(threshold), loss=test_loss
    )
    row.update(
        {
            "fold": int(fold),
            "model": spec.key,
            "condition_label": f"{spec.display_name} local_epochs=5 class-weighted CE normal sampling",
            "algorithm": "fedavg",
            "local_epochs": 5,
            "mu": 0.0,
            "threshold": float(threshold),
            "auc": float(metrics["auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "bacc": float(metrics["bacc"]),
            "acc": float(metrics["acc"]),
            "sensitivity": float(metrics["sensitivity"]),
            "specificity": float(metrics["specificity"]),
            "test_loss": test_loss,
            "output_dir": str(out_dir),
        }
    )
    row.update(count_predictions(pred_df, threshold))
    return row


def write_fold_runtime(fold_dir: Path, row: dict, start_wall: str, end_wall: str, wall_seconds: float) -> None:
    pd.DataFrame(
        [
            {
                "fold": int(row["fold"]),
                "start_timestamp": start_wall,
                "end_timestamp": end_wall,
                "runtime_seconds": float(row.get("seconds", wall_seconds)),
                "runtime_hours": float(row.get("seconds", wall_seconds)) / 3600.0,
                "wall_seconds_this_invocation": float(wall_seconds),
            }
        ]
    ).to_csv(fold_dir / "runtime.tsv", sep="\t", index=False)


def available_fold_results(spec: ModelSpec, threshold: float) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict]]:
    rows = []
    predictions = []
    fold_metric_rows = []
    for fold in range(5):
        fold_dir = spec.out_dir / f"fold_{fold}"
        pred_path = fold_dir / "global_test_predictions.tsv"
        result_path = fold_dir / "fold_result.tsv"
        if not pred_path.exists() or not result_path.exists():
            continue
        pred_df = pd.read_csv(pred_path, sep="\t")
        result = pd.read_csv(result_path, sep="\t").iloc[0].to_dict()
        result = normalized_result_row(spec, result, fold, pred_df, threshold, spec.out_dir)
        rows.append(result)
        predictions.append(pred_df)
        fold_rows = metric_rows_from_predictions(pred_df, fold, "global_cv_test", group_cols=("institution", "site"))
        if fold_rows:
            fold_rows[0].update(
                {
                    "auc": result["auc"],
                    "pr_auc": result["pr_auc"],
                    "bacc": result["bacc"],
                    "acc": result["acc"],
                    "sensitivity": result["sensitivity"],
                    "specificity": result["specificity"],
                    "loss": result["test_loss"],
                    "seconds": result.get("seconds", float("nan")),
                }
            )
        fold_metric_rows.extend(fold_rows)
    return pd.DataFrame(rows), predictions, fold_metric_rows


def write_mean_std(fold_results: pd.DataFrame, spec: ModelSpec, out_dir: Path) -> pd.DataFrame:
    row = {
        "model": spec.display_name,
        "threshold": 0.50,
        "n_folds": int(fold_results["fold"].nunique()) if not fold_results.empty else 0,
        "folds": (
            ",".join(str(int(f)) for f in sorted(pd.to_numeric(fold_results["fold"]).unique()))
            if not fold_results.empty
            else ""
        ),
    }
    for metric in METRICS:
        values = (
            pd.to_numeric(fold_results[metric], errors="coerce") if metric in fold_results else pd.Series(dtype=float)
        )
        row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
        row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else float("nan")
    for col in COUNT_COLS:
        values = pd.to_numeric(fold_results[col], errors="coerce") if col in fold_results else pd.Series(dtype=float)
        row[f"{col}_sum"] = float(values.sum()) if not values.empty else 0.0
        row[f"{col}_mean"] = float(values.mean()) if not values.empty else float("nan")
    seconds = pd.to_numeric(fold_results.get("seconds", pd.Series(dtype=float)), errors="coerce")
    row["runtime_seconds_sum"] = float(seconds.sum()) if not seconds.empty else 0.0
    row["runtime_hours_sum"] = float(seconds.sum() / 3600.0) if not seconds.empty else 0.0
    summary = pd.DataFrame([row])
    summary.to_csv(out_dir / "mean_std_summary.tsv", sep="\t", index=False)
    return summary


def refresh_summaries(spec: ModelSpec, threshold: float) -> pd.DataFrame:
    fold_results, predictions, fold_rows = available_fold_results(spec, threshold)
    if fold_results.empty:
        return pd.DataFrame()
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    fold_results.sort_values("fold").to_csv(spec.out_dir / "fold_results.tsv", sep="\t", index=False)
    if predictions:
        all_predictions = pd.concat(predictions, ignore_index=True)
        all_predictions.to_csv(spec.out_dir / "all_fold_predictions.tsv", sep="\t", index=False)
        all_predictions.to_csv(spec.out_dir / "threshold_ready_predictions.tsv", sep="\t", index=False)
        summarize_outputs(predictions, fold_rows, spec.out_dir, "global_cv_test")
    return write_mean_std(fold_results, spec, spec.out_dir)
