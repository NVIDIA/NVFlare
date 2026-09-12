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

"""Evaluate one completed CIFAR-10 federation with a common post-training protocol.

Every method is evaluated from the final server checkpoint on:

- the full CIFAR-10 test set (``global_accuracy``); and
- the same deterministic site-specific test partitions
  (``client_accuracies`` and ``worst_client_accuracy``).

Using one evaluator for all methods avoids comparing adaptive/FedCE client-side
metrics against baseline clients evaluated on a different test set. For adaptive
runs, federation-level activation telemetry persisted in the final checkpoint is
returned alongside accuracy so fallback-only partial-participation runs are
visible in the evidence.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
PROJECT_SRC = PROJECT_DIR / "src"
CIFAR_SRC = REPO_ROOT / "examples" / "advanced" / "cifar10" / "pt" / "src"
for path in (str(PROJECT_SRC), str(CIFAR_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adaptive_hetero.nvflare_aggregator import AdaptiveMetaKey  # noqa: E402
from data.cifar10_data_utils import CIFAR10_ROOT  # noqa: E402
from model import ModerateCNN  # noqa: E402

ADAPTIVE_TELEMETRY_KEYS = (
    AdaptiveMetaKey.AGGREGATION_ROUNDS,
    AdaptiveMetaKey.ACTIVE_ROUNDS,
    AdaptiveMetaKey.ACTIVATION_RATE,
    AdaptiveMetaKey.MEAN_ACTIVE_BLEND_FACTOR,
    AdaptiveMetaKey.MAX_OBSERVED_BLEND_FACTOR,
    AdaptiveMetaKey.COHORT_CHANGE_COUNT,
)


def find_server_checkpoint(workspace: str) -> Path:
    """Find the final server checkpoint produced by an NVFlare simulation."""

    workspace_path = Path(workspace).resolve()
    server_dir = workspace_path / "server" / "simulate_job" / "app_server"
    # Prefer final-round model for an identical endpoint across methods. Best
    # checkpoints can depend on method-specific client metrics/model selection.
    for name in ("FL_global_model.pt", "best_FL_global_model.pt"):
        candidate = server_dir / name
        if candidate.is_file():
            return candidate

    candidates = [
        path
        for path in workspace_path.rglob("*.pt")
        if "server" in path.parts and path.name in {"FL_global_model.pt", "best_FL_global_model.pt"}
    ]
    if not candidates:
        raise FileNotFoundError(f"no server CIFAR checkpoint found under {workspace_path}")
    candidates.sort(key=lambda path: (path.name != "FL_global_model.pt", str(path)))
    return candidates[0]


def _checkpoint_payload(checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"unsupported checkpoint format in {checkpoint_path}")
    return checkpoint


def checkpoint_state_dict(checkpoint_path: str | Path) -> dict:
    """Load the model state dict from NVFlare's PyTorch checkpoint format."""

    checkpoint = _checkpoint_payload(checkpoint_path)
    if isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise ValueError(f"unsupported checkpoint format in {checkpoint_path}")


def checkpoint_meta(checkpoint_path: str | Path) -> dict:
    """Return persisted NVFlare model metadata when available."""

    checkpoint = _checkpoint_payload(checkpoint_path)
    meta = checkpoint.get("meta_props")
    return dict(meta) if isinstance(meta, dict) else {}


def _test_dataset(download: bool = False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    return datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=download, transform=transform)


def _accuracy(model: torch.nn.Module, dataset, batch_size: int, num_workers: int, device: torch.device) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if len(dataset) == 0:
        raise ValueError("evaluation dataset must contain at least one example")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise ValueError("evaluation loader produced no examples")
    return correct / total


def evaluate_workspace(
    workspace: str,
    eval_idx_root: str,
    n_clients: int,
    batch_size: int = 256,
    num_workers: int = 0,
    device_name: str | None = None,
) -> dict:
    if n_clients < 1:
        raise ValueError("n_clients must be positive")

    checkpoint_path = find_server_checkpoint(workspace)
    checkpoint_metadata = checkpoint_meta(checkpoint_path)
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = ModerateCNN()
    model.load_state_dict(checkpoint_state_dict(checkpoint_path), strict=True)
    model.to(device)

    full_test = _test_dataset(download=False)
    global_accuracy = _accuracy(model, full_test, batch_size, num_workers, device)

    client_accuracies = {}
    client_counts = {}
    for site_index in range(1, n_clients + 1):
        site_name = f"site-{site_index}"
        index_path = Path(eval_idx_root) / f"{site_name}.npy"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing evaluation indices for {site_name}: {index_path}")
        indices = np.load(index_path).astype(np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"evaluation indices for {site_name} must be a non-empty vector")
        if np.any(indices < 0) or np.any(indices >= len(full_test)):
            raise ValueError(f"evaluation indices for {site_name} are outside the CIFAR-10 test set")
        local_accuracy = _accuracy(model, Subset(full_test, indices.tolist()), batch_size, num_workers, device)
        client_accuracies[site_name] = local_accuracy
        client_counts[site_name] = int(indices.size)

    values = np.asarray(list(client_accuracies.values()), dtype=np.float64)
    telemetry = {key: checkpoint_metadata[key] for key in ADAPTIVE_TELEMETRY_KEYS if key in checkpoint_metadata}
    return {
        "checkpoint": str(checkpoint_path),
        "global_accuracy": float(global_accuracy),
        "mean_client_accuracy": float(values.mean()),
        "worst_client_accuracy": float(values.min()),
        "best_client_accuracy": float(values.max()),
        "client_accuracy_gap": float(values.max() - values.min()),
        "client_accuracies": client_accuracies,
        "client_eval_counts": client_counts,
        "adaptive_telemetry": telemetry,
    }


def main(args):
    metrics = evaluate_workspace(
        workspace=os.path.abspath(args.workspace),
        eval_idx_root=os.path.abspath(args.eval_idx_root),
        n_clients=args.n_clients,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--eval_idx_root", required=True)
    parser.add_argument("--n_clients", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
