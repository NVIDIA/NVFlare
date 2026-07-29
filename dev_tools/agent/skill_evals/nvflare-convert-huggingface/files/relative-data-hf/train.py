# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from model import MODEL_NAME, create_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

DEFAULT_DATA_DIR = Path("data")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "valid.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = dataset.map(lambda row: tokenizer(row["text"], truncation=True), batched=True)
    trainer = Trainer(
        model=create_model(),
        args=TrainingArguments(
            output_dir="outputs",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            report_to=[],
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
