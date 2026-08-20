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

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path

# Keep any transitive Matplotlib cache out of environments with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import torch
from model import SmilesCNN
from torch import nn
from torch.utils.data import DataLoader, Dataset

SOURCE_DIR = Path(__file__).resolve().parent
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SMILES_CHARACTERS = "#()+-./0123456789=@BCFHINOPS[]clnors"
VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1}
VOCAB.update({character: index + 2 for index, character in enumerate(SMILES_CHARACTERS)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small SMILES CNN on synthetic mutagenicity records.")
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DIR / "data")
    parser.add_argument("--out-dir", type=Path, default=SOURCE_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_smiles(smiles: str, max_length: int) -> torch.Tensor:
    token_ids = torch.full((max_length,), VOCAB[PAD_TOKEN], dtype=torch.long)
    unknown_id = VOCAB[UNK_TOKEN]
    for index, character in enumerate(smiles[:max_length]):
        token_ids[index] = VOCAB.get(character, unknown_id)
    return token_ids


class SmilesDataset(Dataset):
    def __init__(self, path: Path, max_length: int):
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []
        with path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or not {"smiles", "label"}.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain smiles and label columns")
            for row in reader:
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError(f"{path} contains a non-binary label: {label}")
                self.records.append(
                    (encode_smiles(row["smiles"], max_length), torch.tensor(label, dtype=torch.float32))
                )
        if not self.records:
            raise ValueError(f"{path} contains no records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.records[index]


def make_loader(path: Path, max_length: int, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = SmilesDataset(path, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def binary_auroc(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
    positive_scores = probabilities[labels == 1]
    negative_scores = probabilities[labels == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return float("nan")
    comparisons = (positive_scores[:, None] > negative_scores).float()
    ties = (positive_scores[:, None] == negative_scores).float()
    return float((comparisons + 0.5 * ties).mean().item())


def metric_is_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None or current["auroc"] > best["auroc"]:
        return True
    return current["auroc"] == best["auroc"] and current["loss"] < best["loss"]


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for token_ids, labels in loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            logits = model(token_ids)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            all_labels.append(labels.cpu())
            all_probabilities.append(torch.sigmoid(logits).cpu())

    labels = torch.cat(all_labels)
    probabilities = torch.cat(all_probabilities)
    predictions = (probabilities >= 0.5).float()
    return {
        "loss": total_loss / len(labels),
        "accuracy": float((predictions == labels).float().mean().item()),
        "auroc": binary_auroc(labels, probabilities),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_items = 0
    optimizer_steps = 0
    for token_ids, labels in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(token_ids), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_items += len(labels)
        optimizer_steps += 1
    return total_loss / total_items, optimizer_steps


def get_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.max_length < 7:
        raise SystemExit("--max-length must be at least 7")

    set_seed(args.seed)
    device = get_device(args.device)
    train_loader = make_loader(args.data_dir / "train.csv", args.max_length, args.batch_size, True, args.seed)
    valid_loader = make_loader(args.data_dir / "valid.csv", args.max_length, args.batch_size, False, args.seed)
    test_loader = make_loader(args.data_dir / "test.csv", args.max_length, args.batch_size, False, args.seed)

    model_args = {
        "vocab_size": len(VOCAB),
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "dropout": args.dropout,
    }
    model = SmilesCNN(**model_args).to(device)
    train_labels = torch.tensor([int(label.item()) for _, label in train_loader.dataset])
    positive_count = int((train_labels == 1).sum().item())
    negative_count = len(train_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("training data must contain both labels")
    pos_weight = torch.tensor(negative_count / positive_count, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "best_model.pt"
    history = []
    best_valid = None
    total_optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, optimizer_steps = train_one_epoch(model, train_loader, criterion, optimizer, device)
        total_optimizer_steps += optimizer_steps
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "optimizer_steps": optimizer_steps, "valid": valid_metrics}
        )
        if metric_is_better(valid_metrics, best_valid):
            best_valid = valid_metrics
            torch.save({"model_state": model.state_dict(), "model_args": model_args, "vocab": VOCAB}, checkpoint_path)
        print(
            f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={valid_metrics['loss']:.4f} | val_accuracy={valid_metrics['accuracy']:.4f} | "
            f"val_auroc={valid_metrics['auroc']:.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "best_valid": best_valid,
                "test": test_metrics,
                "history": history,
                "optimizer_steps": total_optimizer_steps,
            },
            metrics_file,
            indent=2,
        )
    print(
        f"best validation | loss={best_valid['loss']:.4f} | "
        f"accuracy={best_valid['accuracy']:.4f} | auroc={best_valid['auroc']:.4f}"
    )
    print(
        f"test            | loss={test_metrics['loss']:.4f} | "
        f"accuracy={test_metrics['accuracy']:.4f} | auroc={test_metrics['auroc']:.4f}"
    )
    try:
        checkpoint_display = checkpoint_path.resolve().relative_to(SOURCE_DIR.parent)
    except ValueError:
        checkpoint_display = checkpoint_path
    print(f"saved           | {checkpoint_display} | optimizer_steps={total_optimizer_steps}")


if __name__ == "__main__":
    main()
