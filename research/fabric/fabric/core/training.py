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

"""Feature loading, local training, and patient-level evaluation for FABRIC."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from .mil_models import build_mil_model

METRIC_COLUMNS = [
    "acc",
    "recall",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "bacc",
    "auc",
    "pr_auc",
    "youden_j",
    "loss",
    "n",
    "pos",
    "neg",
    "threshold",
]


PREDICTION_METADATA = [
    "patient_id",
    "institution",
    "site",
    "tumor_grade",
    "num_slides",
    "slide_ids",
    "feature_paths",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    gpu = int(getattr(args, "gpu", 0))
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    return torch.device(f"cuda:{gpu}")


def read_feature_shape(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            key = next((k for k in ["features", "feature", "feat", "x"] if k in handle), None)
            if key is None:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError(f"No datasets in {path}")
                key = keys[0]
            shape = tuple(handle[key].shape)
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ["features", "feature", "feat", "x"]:
                if key in obj:
                    obj = obj[key]
                    break
        shape = tuple(torch.as_tensor(obj).shape)
    elif suffix == ".npy":
        shape = tuple(np.load(path, mmap_mode="r").shape)
    else:
        raise ValueError(f"Unsupported feature file type: {path}")
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) != 2:
        raise ValueError(f"Expected 2D features, got {shape} from {path}")
    return int(shape[0]), int(shape[1])


def normalize_feature_tensor(value: object, path: Path, row_indices: np.ndarray | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if row_indices is not None:
        tensor = tensor[torch.as_tensor(row_indices, dtype=torch.long)]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D features from {path}, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def read_feature_array(path: str | Path, row_indices: np.ndarray | None = None) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            key = next((k for k in ["features", "feature", "feat", "x"] if k in handle), None)
            if key is None:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError(f"No datasets in {path}")
                key = keys[0]
            dataset = handle[key]
            if row_indices is None:
                return normalize_feature_tensor(dataset[:], path)
            order = np.argsort(row_indices)
            sorted_indices = np.asarray(row_indices, dtype=np.int64)[order]
            values = dataset[sorted_indices]
            reverse = np.argsort(order)
            return normalize_feature_tensor(values[reverse], path)
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ["features", "feature", "feat", "x"]:
                if key in obj:
                    obj = obj[key]
                    break
        return normalize_feature_tensor(obj, path, row_indices=row_indices)
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if row_indices is not None:
            arr = arr[row_indices]
        return normalize_feature_tensor(arr, path)
    raise ValueError(f"Unsupported feature file type: {path}")


def select_instance_indices(total_instances: int, max_instances: int, training: bool) -> np.ndarray | None:
    if max_instances <= 0 or total_instances <= max_instances:
        return None
    if training:
        return torch.randperm(total_instances)[:max_instances].numpy().astype(np.int64, copy=False)
    return np.linspace(0, total_instances - 1, max_instances, dtype=np.int64)


def read_patient_bag(paths: Sequence[str], max_instances: int, training: bool, patient_id: str) -> torch.Tensor:
    paths = list(paths)
    if not paths:
        raise ValueError(f"No feature paths for {patient_id}")
    shapes = [read_feature_shape(Path(path)) for path in paths]
    dims = [dim for _count, dim in shapes]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent feature dims for {patient_id}: {dims}")
    counts = np.asarray([count for count, _dim in shapes], dtype=np.int64)
    selected = select_instance_indices(int(counts.sum()), max_instances=max_instances, training=training)
    if selected is None:
        return torch.cat([read_feature_array(path) for path in paths], dim=0)

    dim = int(dims[0])
    bag = torch.empty((len(selected), dim), dtype=torch.float32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for slide_idx, path in enumerate(paths):
        start = int(offsets[slide_idx])
        stop = int(offsets[slide_idx + 1])
        positions = np.flatnonzero((selected >= start) & (selected < stop))
        if positions.size == 0:
            continue
        local_indices = selected[positions] - start
        rows = read_feature_array(path, row_indices=local_indices)
        bag[torch.as_tensor(positions, dtype=torch.long)] = rows
    return bag


class PatientFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_instances: int, training: bool):
        self.df = df.reset_index(drop=True)
        self.max_instances = int(max_instances)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[index]
        paths = [item for item in str(row["feature_paths"]).split(";") if item]
        bag = read_patient_bag(paths, self.max_instances, self.training, str(row["patient_id"]))
        label = torch.tensor(int(row["recurrence_label"]), dtype=torch.long)
        return bag, label, str(row["patient_id"])


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_patient_bags(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[list[torch.Tensor], torch.Tensor, list[str]]:
    bags, labels, patient_ids = zip(*batch)  # noqa: B905
    return list(bags), torch.stack(list(labels), dim=0), list(patient_ids)


def make_loader(df: pd.DataFrame, args: argparse.Namespace, seed: int, training: bool, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PatientFeatureDataset(df, max_instances=int(args.max_instances), training=training),
        batch_size=int(args.batch_size),
        shuffle=training,
        num_workers=int(args.num_workers),
        collate_fn=collate_patient_bags,
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
        generator=generator if training else None,
        pin_memory=pin_memory,
    )


def forward_bags(model: nn.Module, bags: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    logits = []
    for bag in bags:
        output = model(bag.unsqueeze(0).to(device, non_blocking=True))
        value = output["logits"] if isinstance(output, dict) else output
        logits.append(value.squeeze(0))
    return torch.stack(logits, dim=0)


def build_model(model_name: str, input_dim: int, args: argparse.Namespace) -> nn.Module:
    return build_mil_model(
        model_name=model_name,
        input_dim=int(input_dim),
        n_classes=2,
        embed_dim=int(args.embed_dim),
        attn_dim=int(args.attn_dim),
        dropout=float(args.dropout),
        dtfd_pseudo_bags=int(args.dtfd_pseudo_bags),
        dtfd_top_k=getattr(args, "dtfd_top_k", 1),
        dtfd_eval_group_seed=getattr(args, "dtfd_eval_group_seed", 42),
    )


def build_criterion(df: pd.DataFrame, device: torch.device) -> nn.Module:
    labels = df["recurrence_label"].astype(int)
    neg = int((labels == 0).sum())
    pos = int((labels == 1).sum())
    pos_weight = float(neg / pos) if pos > 0 and neg > 0 else 1.0
    return nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device))


def compute_metrics(
    labels: Sequence[int], probs: Sequence[float], threshold: float, loss: float = float("nan")
) -> dict:
    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    preds = (probs_arr >= float(threshold)).astype(int)
    out = {key: float("nan") for key in METRIC_COLUMNS}
    out["loss"] = float(loss)
    out["n"] = int(len(labels_arr))
    out["pos"] = int((labels_arr == 1).sum())
    out["neg"] = int((labels_arr == 0).sum())
    out["threshold"] = float(threshold)
    if len(labels_arr) == 0:
        return out
    out["acc"] = float(accuracy_score(labels_arr, preds))
    out["precision"] = float(precision_score(labels_arr, preds, zero_division=0))
    out["recall"] = float(recall_score(labels_arr, preds, zero_division=0))
    out["sensitivity"] = out["recall"]
    out["f1"] = float(f1_score(labels_arr, preds, zero_division=0))
    out["bacc"] = float(balanced_accuracy_score(labels_arr, preds))
    tn, fp, fn, tp = confusion_matrix(labels_arr, preds, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    if len(np.unique(labels_arr)) == 2:
        out["auc"] = float(roc_auc_score(labels_arr, probs_arr))
        out["pr_auc"] = float(average_precision_score(labels_arr, probs_arr))
        fpr, tpr, _thresholds = roc_curve(labels_arr, probs_arr)
        out["youden_j"] = float(np.max(tpr - fpr))
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.amp.GradScaler,
) -> dict:
    model.train(True)
    total_loss = 0.0
    total_count = 0
    autocast_enabled = bool(args.amp and device.type == "cuda")
    use_topk = getattr(args, "model_variant", "pooling") == "topk"
    patient_loss_total = pseudo_loss_total = 0.0
    if use_topk:
        from .topk_mil import forward_topk_bags
    for bags, labels, _patient_ids in loader:
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            if use_topk:
                logits, pseudo_loss = forward_topk_bags(model, bags, labels, criterion, device)
                patient_loss = criterion(logits, labels)
                loss = patient_loss + float(args.dtfd_pseudo_loss_weight) * pseudo_loss
            else:
                logits = forward_bags(model, bags, device)
                loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
        if use_topk:
            patient_loss_total += float(patient_loss.detach().cpu().item()) * batch_size
            pseudo_loss_total += float(pseudo_loss.detach().cpu().item()) * batch_size
    if total_count == 0:
        raise ValueError("Training loader yielded no patients; check the dataset, sampler, and batch settings.")
    log = {"loss": total_loss / total_count, "n": int(total_count)}
    if use_topk:
        log.update(patient_loss=patient_loss_total / total_count, pseudo_loss=pseudo_loss_total / total_count)
    return log


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    df: pd.DataFrame,
    criterion_df: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    loader = make_loader(df, args, seed=seed, training=False, pin_memory=device.type == "cuda")
    criterion = build_criterion(criterion_df, device)
    model.eval()
    patient_ids: list[str] = []
    labels_all: list[int] = []
    probs_all: list[float] = []
    total_loss = 0.0
    total_count = 0
    autocast_enabled = bool(args.amp and device.type == "cuda")
    for bags, labels, batch_patient_ids in loader:
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = forward_bags(model, bags, device)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
        patient_ids.extend(batch_patient_ids)
        labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
        probs_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
    if total_count == 0:
        raise ValueError("Evaluation loader yielded no patients; check the dataset, sampler, and batch settings.")
    metrics = compute_metrics(labels_all, probs_all, threshold=float(args.threshold), loss=total_loss / total_count)
    pred_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "recurrence_label": labels_all,
            "prob_recurrence": probs_all,
            "pred": [int(prob >= float(args.threshold)) for prob in probs_all],
            "threshold": float(args.threshold),
        }
    )
    pred_df = pred_df.merge(df[PREDICTION_METADATA], on="patient_id", how="left", validate="one_to_one")
    return metrics, pred_df


def metric_rows_from_predictions(
    pred_df: pd.DataFrame, fold: int | str, test_set: str, group_cols: Sequence[str] = ("institution",)
) -> list[dict]:
    rows = []
    groups = [("OVERALL", pred_df)]
    for col in group_cols:
        for value, sub in pred_df.groupby(col, sort=True):
            groups.append((f"{col}:{value}", sub))
    for group_name, sub in groups:
        metrics = compute_metrics(sub["recurrence_label"], sub["prob_recurrence"], float(sub["threshold"].iloc[0]))
        row = {
            "fold": fold,
            "test_set": test_set,
            "group": group_name,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize_outputs(all_predictions: list[pd.DataFrame], fold_rows: list[dict], out_dir: Path, test_set: str) -> None:
    if not all_predictions:
        return
    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df.to_csv(out_dir / "all_fold_predictions.tsv", sep="\t", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics.tsv", sep="\t", index=False)

    summary_rows = []
    metric_cols = [col for col in METRIC_COLUMNS if col not in {"n", "pos", "neg", "threshold"}]
    for group, sub in fold_df.groupby("group", sort=True):
        row = {"test_set": test_set, "group": group, "n_folds": int(sub["fold"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
            row[f"{col}_std"] = float(pd.to_numeric(sub[col], errors="coerce").std())
        row["n_sum_over_folds"] = int(pd.to_numeric(sub["n"], errors="coerce").sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.tsv", sep="\t", index=False)

    pooled_rows = metric_rows_from_predictions(
        pred_df, fold="pooled", test_set=test_set, group_cols=("institution", "site")
    )
    pd.DataFrame(pooled_rows).to_csv(out_dir / "pooled_metrics.tsv", sep="\t", index=False)
