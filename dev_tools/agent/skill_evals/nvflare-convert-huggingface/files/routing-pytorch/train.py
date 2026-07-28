# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


def train(model_name, batches):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for batch in batches:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"])
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
