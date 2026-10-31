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

"""FedMoSA data: the same slices, transforms and batching as the original implementation.

Slices come from the files built by the original code (``system/prepare_data.py``) in ``<data-dir>``, where
site-<k+1> is client k: ``slices_<k>_train_<id>_volumes.pkl`` holds every slice of every training volume
([(modality, [image], [mask]), ...]); ``slices_<k>_<split>_<id>.pkl`` holds the val / test slices
([(image, mask, modality), ...]). Images are uint8 HxW, masks uint8 HxW in {0, 255}.

Every local epoch trains on a fresh random 25% of every training volume's slices, with the same seed as the
original code (dataset, run, client, round, epoch), so both implementations draw the same subsets.

Batches hold one modality (the model routes a whole batch through one expert). Every pass shuffles each
modality, trims it to full batches and alternates modalities, and every slice gets a random 90/180/270 degree
rotation, exactly as in the original client.
"""

import hashlib
import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

TRAIN_FRACTION = 0.25  # share of every training volume's slices drawn afresh in each epoch

transform_x = transforms.Compose(
    [transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)
transform_y = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)


def set_seed(seed=32):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def slice_file(data_dir, client_idx, split, dataset_id="0"):
    if split == "train":
        return os.path.join(data_dir, f"slices_{client_idx}_train_{dataset_id}_volumes.pkl")
    return os.path.join(data_dir, f"slices_{client_idx}_{split}_{dataset_id}.pkl")


def train_seed(dataset, dataset_id, client_idx, rnd, epoch):
    """Seed of one client's training subset in one epoch of one round (identical to the original code)."""
    key = f"{dataset}|{dataset_id}|{client_idx}|{rnd}|{epoch}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def epoch_subset(volumes, seed, fraction=TRAIN_FRACTION):
    """A fresh random int(n * fraction) slices of every training volume -> [(image, mask, modality), ...]."""
    rng = random.Random(seed)
    out = []
    for modality, images, masks in volumes:
        for i in rng.sample(range(len(images)), int(len(images) * fraction)):
            out.append((images[i], masks[i], modality))
    return out


def subset_counts(volumes, fraction=TRAIN_FRACTION):
    """Slices per modality in every epoch's subset (the same in every epoch)."""
    counts = {}
    for modality, images, _ in volumes:
        counts[modality] = counts.get(modality, 0) + int(len(images) * fraction)
    return counts


def load_split(data_dir, client_idx, split, dataset_id="0"):
    """train: the training volumes; val / test: [(image, mask, modality), ...]; None if the split does not exist."""
    path = slice_file(data_dir, client_idx, split, dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_image(image, mask, rotate=True):
    x = Image.fromarray(image).convert("L")
    y = Image.fromarray(mask).convert("L")
    if rotate:
        angle = random.choice([90, 180, 270])
        x, y = x.rotate(angle), y.rotate(angle)
    return transform_x(x).repeat(3, 1, 1), transform_y(y)


def modality_counts(data):
    counts = {}
    for _, _, m in data:
        counts[m] = counts.get(m, 0) + 1
    return counts


def _alternating_batches(by_modality, batch_size):
    """Full batches only; the modality with most batches first, then the others in order, round-robin."""
    batched = OrderedDict((m, ds[: (len(ds) // batch_size) * batch_size]) for m, ds in by_modality.items())
    n_batches = {m: len(ds) // batch_size for m, ds in batched.items()}
    if not n_batches:
        return []
    order = sorted(n_batches, key=lambda m: n_batches[m], reverse=True)
    ptr = {m: 0 for m in n_batches}
    out = []
    while True:
        added = False
        for m in order:
            if ptr[m] < n_batches[m]:
                out.append(batched[m][ptr[m] * batch_size : (ptr[m] + 1) * batch_size])
                ptr[m] += 1
                added = True
        if not added:
            return out


def batches(data, batch_size, threshold):
    """Yields (images, masks, modality); masks are binarized with ``>= 0.5`` (train) or ``> 0.5`` (eval)."""
    by_modality = {}
    for item in data:
        by_modality.setdefault(item[2], []).append(item)
    for m in by_modality:
        random.shuffle(by_modality[m])
    for batch in _alternating_batches(by_modality, batch_size):
        pairs = [load_image(img, msk) for img, msk, _ in batch]
        x = torch.stack([p[0] for p in pairs])
        y = torch.stack([p[1] for p in pairs])
        y = (y >= 0.5).float() if threshold == "ge" else (y > 0.5).float()
        yield x, y, batch[0][2]


def make_synthetic(data_dir, sites_modalities, n_train=16, n_test=4, size=64, seed=0):
    """Tiny random-ellipse slices in the same file formats, for smoke tests (no real images).
    Training: one volume of n_train slices per modality; val / test: n_test slices per modality."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size

    def one_slice():
        cy, cx, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.25)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2).astype(np.uint8) * 255
        img = np.clip(80 + 0.5 * mask + 25 * rng.standard_normal((size, size)), 0, 255).astype(np.uint8)
        return img, mask

    os.makedirs(data_dir, exist_ok=True)
    for k, mods in enumerate(sites_modalities):
        volumes = []
        for m in mods:
            pairs = [one_slice() for _ in range(n_train)]
            volumes.append((m, [p[0] for p in pairs], [p[1] for p in pairs]))
        with open(slice_file(data_dir, k, "train"), "wb") as f:
            pickle.dump(volumes, f)
        for split in ("val", "test"):
            data = [(*one_slice(), m) for m in mods for _ in range(n_test)]
            with open(slice_file(data_dir, k, split), "wb") as f:
                pickle.dump(data, f)
