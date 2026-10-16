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

"""Local client optimization and fold-level result records."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from .aggregation import copy_state_dict
from .training import build_criterion, build_model, make_loader
from .training import resolve_device as resolve_device  # noqa: F401 - Re-exported for the FLARE client.
from .training import set_seed, train_one_epoch


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    algorithm: str
    dirname: str
    mu: float
    source_fold0_dir: Path


def result_row(
    condition: Condition, fold: int, out_dir: Path, metrics: dict, pred_df: pd.DataFrame, seconds: float
) -> dict:
    labels = pred_df["recurrence_label"].astype(int)
    preds = pred_df["pred"].astype(int)
    return {
        "fold": int(fold),
        "model": condition.key,
        "condition_label": condition.label,
        "algorithm": condition.algorithm,
        "local_epochs": 3,
        "mu": float(condition.mu),
        "threshold": float(pred_df["threshold"].iloc[0]) if "threshold" in pred_df else 0.5,
        "auc": float(metrics.get("auc", float("nan"))),
        "bacc": float(metrics.get("bacc", float("nan"))),
        "acc": float(metrics.get("acc", float("nan"))),
        "sensitivity": float(metrics.get("sensitivity", float("nan"))),
        "specificity": float(metrics.get("specificity", float("nan"))),
        "test_loss": float(metrics.get("loss", float("nan"))),
        "predicted_positives": int(preds.sum()),
        "true_positives_detected": int(((labels == 1) & (preds == 1)).sum()),
        "true_positives_total": int((labels == 1).sum()),
        "false_positives": int(((labels == 0) & (preds == 1)).sum()),
        "false_negatives": int(((labels == 1) & (preds == 0)).sum()),
        "n_test": int(len(pred_df)),
        "output_dir": str(out_dir),
        "seconds": float(seconds),
    }


def train_local_client_final(
    condition: Condition,
    global_state: dict[str, torch.Tensor],
    client_df: pd.DataFrame,
    model_name: str,
    input_dim: int,
    args: SimpleNamespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    if condition.algorithm != "fedavg":
        raise ValueError("FABRIC supports FedAvg client training")
    set_seed(seed)
    model = build_model(model_name, input_dim, args).to(device)
    model.load_state_dict(global_state)
    criterion = build_criterion(client_df, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    loader = make_loader(client_df, args, seed=seed, training=True, pin_memory=device.type == "cuda")
    epoch_logs = []
    start = time.perf_counter()
    for epoch in range(1, int(args.local_epochs) + 1):
        log = train_one_epoch(model, loader, criterion, optimizer, device, args, scaler)
        log["local_epoch"] = epoch
        epoch_logs.append(log)
    log_df = pd.DataFrame(epoch_logs)
    out = {
        "loss": float(log_df["loss"].mean()) if not log_df.empty else float("nan"),
        "train_patients": int(len(client_df)),
        "seconds": time.perf_counter() - start,
    }
    if getattr(args, "model_variant", "pooling") == "topk":
        out.update({key: float(log_df[key].mean()) for key in ("patient_loss", "pseudo_loss")})
    return copy_state_dict(model.state_dict()), out
