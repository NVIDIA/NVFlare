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

"""2D slice datasets for FedMoSA.

Each site reads <data_root>/<site>/{train,test}.csv with columns image,mask,modality. Image and mask are .npy
2D slices (image already intensity-normalized to [0, 1], mask binary). Batches are grouped by modality because
the model routes a whole batch through one modality expert.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_SIZE = 1024
MASK_SIZE = 256


class SliceDataset(Dataset):
    def __init__(self, rows, train=True):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img_path, mask_path, modality = self.rows[i]
        img = torch.from_numpy(np.load(img_path).astype(np.float32))[None, None]
        mask = torch.from_numpy((np.load(mask_path) > 0).astype(np.float32))[None, None]
        img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask, size=(MASK_SIZE, MASK_SIZE), mode="nearest")[0]
        if self.train and random.random() < 0.5:
            k = random.randint(1, 3)
            img, mask = torch.rot90(img, k, (1, 2)), torch.rot90(mask, k, (1, 2))
        img = ((img - 0.5) / 0.5).repeat(3, 1, 1)
        return img, mask, modality


def read_rows(data_root, site, split):
    with open(os.path.join(data_root, site, f"{split}.csv")) as f:
        return [(r["image"], r["mask"], r["modality"]) for r in csv.DictReader(f)]


def modality_batches(dataset, batch_size, shuffle=True):
    """Yields (images, masks, modality) with one modality per batch."""
    by_mod = {}
    for i, (_, _, m) in enumerate(dataset.rows):
        by_mod.setdefault(m, []).append(i)
    chunks = []
    for m, idx in by_mod.items():
        if shuffle:
            random.shuffle(idx)
        chunks += [(m, idx[j : j + batch_size]) for j in range(0, len(idx), batch_size)]
    if shuffle:
        random.shuffle(chunks)
    for m, idx in chunks:
        items = [dataset[i] for i in idx]
        yield torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]), m


def modality_counts(rows):
    counts = {}
    for _, _, m in rows:
        counts[m] = counts.get(m, 0) + 1
    return counts


def make_synthetic(data_root, sites_modalities, n_train=4, n_test=2, size=64, seed=0):
    """Tiny random-ellipse dataset for smoke tests (no real images)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    for site, mods in sites_modalities.items():
        os.makedirs(os.path.join(data_root, site), exist_ok=True)
        for split, n in (("train", n_train), ("test", n_test)):
            rows = []
            for m in mods:
                for k in range(n):
                    cy, cx, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.25)
                    mask = ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2).astype(np.uint8)
                    img = np.clip(0.3 + 0.5 * mask + 0.1 * rng.standard_normal((size, size)), 0, 1)
                    base = os.path.join(data_root, site, f"{split}_{m}_{k}")
                    np.save(base + "_img.npy", img.astype(np.float32))
                    np.save(base + "_mask.npy", mask)
                    rows.append({"image": base + "_img.npy", "mask": base + "_mask.npy", "modality": m})
            with open(os.path.join(data_root, site, f"{split}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["image", "mask", "modality"])
                w.writeheader()
                w.writerows(rows)
