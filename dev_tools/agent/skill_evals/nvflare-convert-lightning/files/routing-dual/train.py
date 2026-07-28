# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lightning as L
from transformers import Trainer, TrainingArguments


def train(lightning_model, train_loader, hf_model, hf_dataset):
    lightning_trainer = L.Trainer(max_epochs=1)
    lightning_trainer.fit(lightning_model, train_dataloaders=train_loader)

    hf_args = TrainingArguments(output_dir="outputs", report_to=[])
    hf_trainer = Trainer(model=hf_model, args=hf_args, train_dataset=hf_dataset)
    hf_trainer.train()
