# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import Trainer, TrainingArguments


def train(model, train_dataset):
    args = TrainingArguments(output_dir="outputs", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
