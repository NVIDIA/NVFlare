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

"""Small offline data/model; execute the shipped CIFAR client training loops."""

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class TinyCifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(3072, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 10))

    def forward(self, inputs):
        return self.layers(inputs)


class OfflineCifar(Dataset):
    def __init__(self, seed, training):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(8, 3, 32, 32, generator=generator)
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.training:
            image = image + 0.01 * (torch.rand_like(image) + random.random() + np.random.random())
        return image, index % 10


def snapshot(value):
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return torch.as_tensor(value).detach().cpu().clone()
    return value


def main():
    import cifar_client as client
    from cifar_task_fixture import OfflineCifar, TinyCifarModel

    import nvflare.client as flare

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    client.DEVICE = "cpu"
    client.ModerateCNN = TinyCifarModel
    client.create_datasets = lambda site_name, **kwargs: (
        OfflineCifar(731 + int(site_name.rsplit("-", 1)[1]), True),
        OfflineCifar(991, False),
    )
    original_receive, original_send = flare.receive, flare.send
    original_save = client.save_training_state
    folder = Path("cifar_state_evidence")
    folder.mkdir(exist_ok=True)
    current_round = None

    def receive(*args, **kwargs):
        nonlocal current_round
        model = original_receive(*args, **kwargs)
        if flare.is_train():
            current_round = model.current_round
            torch.save(snapshot(model.params), folder / f"{current_round}_input.pt")
        return model

    def save(optimizer, scheduler, current_round, **kwargs):
        original_save(optimizer, scheduler, current_round, **kwargs)
        numpy_rng = np.random.get_state()
        torch.save(
            snapshot(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loaders": [loader.state_dict() for loader in kwargs["data_loaders"]],
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": (numpy_rng[0], numpy_rng[1].astype(np.int64), *numpy_rng[2:]),
                }
            ),
            folder / f"{current_round}_state.pt",
        )

    def send(model, *args, **kwargs):
        if flare.is_train():
            torch.save(
                snapshot(
                    {"diff": model.params, "metrics": model.metrics, "steps": model.meta["NUM_STEPS_CURRENT_ROUND"]}
                ),
                folder / f"{current_round}_output.pt",
            )
        return original_send(model, *args, **kwargs)

    flare.receive, flare.send = receive, send
    client.save_training_state = save
    client.main(
        SimpleNamespace(
            aggregation_epochs=2,
            batch_size=2,
            cosine_lr_eta_min_factor=0.01,
            evaluate_local=True,
            lr=0.02,
            no_lr_scheduler=False,
            num_workers=args.num_workers,
            train_idx_root="unused-offline-test-data",
        )
    )


if __name__ == "__main__":
    main()
