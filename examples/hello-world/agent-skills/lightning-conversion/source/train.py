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
import logging
import os
import tempfile
import warnings
from pathlib import Path

# Keep Matplotlib's transitive cache out of environments with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# These warnings are expected for this intentionally tiny, reusable example.
warnings.filterwarnings("ignore", message=r".*Checkpoint directory .* exists and is not empty.*")
warnings.filterwarnings("ignore", message=r".*does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*")
warnings.filterwarnings("ignore", message=r"GPU available but not used.*")

import pytorch_lightning as pl
import torch
from model import LitSmilesCNN
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

SOURCE_DIR = Path(__file__).resolve().parent
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SMILES_CHARACTERS = "#()+-./0123456789=@BCFHINOPS[]clnors"
VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1}
VOCAB.update({character: index + 2 for index, character in enumerate(SMILES_CHARACTERS)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lightning SMILES CNN on synthetic mutagenicity records.")
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
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    return parser.parse_args()


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


class CompactMetricsPrinter(Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        required = ("train_loss", "val_loss", "val_accuracy", "val_auroc")
        if not all(name in metrics for name in required):
            return
        print(
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} | "
            f"train_loss={float(metrics['train_loss']):.4f} | "
            f"val_loss={float(metrics['val_loss']):.4f} | "
            f"val_accuracy={float(metrics['val_accuracy']):.4f} | "
            f"val_auroc={float(metrics['val_auroc']):.4f}"
        )


def configure_console_output() -> None:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.max_length < 7:
        raise SystemExit("--max-length must be at least 7")

    configure_console_output()
    pl.seed_everything(args.seed, workers=True)
    train_loader = make_loader(args.data_dir / "train.csv", args.max_length, args.batch_size, True, args.seed)
    valid_loader = make_loader(args.data_dir / "valid.csv", args.max_length, args.batch_size, False, args.seed)
    test_loader = make_loader(args.data_dir / "test.csv", args.max_length, args.batch_size, False, args.seed)

    model_args = {
        "vocab_size": len(VOCAB),
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
    }
    model = LitSmilesCNN(**model_args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best_model",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        enable_version_counter=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        callbacks=[checkpoint_callback, CompactMetricsPrinter()],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, ckpt_path="best", verbose=False)[0]
    test_metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best", verbose=False)[0]

    metrics = {
        "best_valid": valid_metrics,
        "test": test_metrics,
        "optimizer_steps": trainer.global_step,
        "model_args": model_args,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(
        f"best validation | loss={valid_metrics['val_loss']:.4f} | "
        f"accuracy={valid_metrics['val_accuracy']:.4f} | auroc={valid_metrics['val_auroc']:.4f}"
    )
    print(
        f"test            | loss={test_metrics['test_loss']:.4f} | "
        f"accuracy={test_metrics['test_accuracy']:.4f} | auroc={test_metrics['test_auroc']:.4f}"
    )
    checkpoint_path = Path(checkpoint_callback.best_model_path)
    try:
        checkpoint_display = checkpoint_path.relative_to(SOURCE_DIR.parent)
    except ValueError:
        checkpoint_display = checkpoint_path
    print(f"saved           | {checkpoint_display} | optimizer_steps={trainer.global_step}")


if __name__ == "__main__":
    main()
