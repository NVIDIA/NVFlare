# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from builder import build_trainer


def train(model, train_dataset):
    trainer = build_trainer(model, train_dataset)
    trainer.train()
