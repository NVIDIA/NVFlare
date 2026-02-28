# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared helpers: seeding, param exchange, generic training loop."""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vqa_soft_score(pred: str, gt_answers: List[str]) -> float:
    """VQA v2 official soft accuracy: min(#matches / 3, 1)."""
    p = pred.strip().lower()
    return min(1.0, sum(1 for a in gt_answers if p == a.strip().lower()) / 3.0)


def maybe_subsample(ds, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples < 0 or max_samples >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def shard_dataset(ds, num_clients: int, site_id: int,
                  alpha: float = 0.0, seed: int = 42,
                  label_key: str = "multiple_choice_answer"):
    """Partition a HuggingFace dataset across clients.

    Args:
        ds: HuggingFace Dataset.
        num_clients: total number of FL clients.
        site_id: this client's index (0-based).
        alpha: Dirichlet concentration.
            alpha <= 0  -> deterministic round-robin (IID baseline).
            alpha > 0   -> Dirichlet non-IID (lower = more skewed).
        seed: random seed for reproducibility.
        label_key: column name used as the label for Dirichlet grouping.
            For VQA datasets this is typically "multiple_choice_answer".
    """
    if alpha <= 0.0:
        # IID round-robin fallback
        return ds.select([i for i in range(len(ds)) if i % num_clients == site_id])

    rng = np.random.default_rng(seed)
    n = len(ds)

    # Build label -> indices mapping
    if label_key in ds.column_names:
        # Map answer strings to integer class ids
        answers = ds[label_key]
        unique = sorted(set(answers))
        label_to_id = {a: i for i, a in enumerate(unique)}
        labels = np.array([label_to_id[a] for a in answers])
    else:
        # Fallback: use index mod 100 as pseudo-label
        labels = np.array([i % 100 for i in range(n)])

    num_classes = int(labels.max()) + 1

    # Dirichlet: for each class, sample a proportion vector over clients
    # proportions[c] = [p_client0, p_client1, ..., p_clientK]
    # sum(proportions[c]) = 1
    proportions = rng.dirichlet([alpha] * num_clients, size=num_classes)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Split class_idx according to proportions
        cumsum = np.cumsum(proportions[c])
        # Convert proportions to index boundaries
        splits = (cumsum[:-1] * len(class_idx)).astype(int)
        chunks = np.split(class_idx, splits)

        for k in range(num_clients):
            client_indices[k].extend(chunks[k].tolist())

    # Sort for deterministic ordering, then return this client's subset
    indices = sorted(client_indices[site_id])
    return ds.select(indices)


def count_trainable_params(model) -> str:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return f"trainable: {t:,} / {a:,} ({100*t/a:.4f}%)"


def get_trainable_params(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_params(model, params: Dict[str, Any], device: str) -> None:
    tmap = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, v in (params or {}).items():
        if n not in tmap:
            continue
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        tmap[n].data.copy_(v.to(device=device, dtype=tmap[n].dtype))


def train_one_epoch(model, dataloader, optimizer, device, grad_accum, backend) -> float:
    """Generic training loop - delegates per-batch loss to backend.train_step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, num_steps = 0.0, len(dataloader)

    for step, batch in enumerate(dataloader, 1):
        loss = backend.train_step(model, batch, device)
        total_loss += loss.item()
        (loss / grad_accum).backward()
        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if num_steps % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_steps, 1)
