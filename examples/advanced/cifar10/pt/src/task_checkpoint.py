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

"""Native training state for the CIFAR clients; CJ owns checkpoint promotion."""

import random

import numpy as np
import torch

import nvflare.client as flare


def restore_training_state(optimizer, scheduler, current_round, *, data_loaders=()):
    info = flare.system_info()
    paths = info.get("task_checkpoint")
    if paths is None or paths["input"] is None:
        return
    state = torch.load(paths["input"], map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["job_id"] != info["job_id"] or state["site_name"] != info["site_name"]:
        raise RuntimeError("Training checkpoint belongs to a different job/site or has an unsupported version")
    previous_round = state["round"]
    if type(previous_round) is not int or not 0 <= previous_round < current_round:
        raise RuntimeError("Training checkpoint must come from an earlier successful round")
    if (state["scheduler"] is None) != (scheduler is None):
        raise RuntimeError("Training checkpoint scheduler configuration changed")
    if len(state["loaders"]) != len(data_loaders):
        raise RuntimeError("Training checkpoint DataLoader configuration changed")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    for loader, loader_state in zip(data_loaders, state["loaders"]):
        loader.load_state_dict(loader_state)
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(state["torch_rng"])


def save_training_state(optimizer, scheduler, current_round, *, data_loaders=()):
    info = flare.system_info()
    paths = info.get("task_checkpoint")
    if paths is None:
        return
    numpy_rng = np.random.get_state()
    torch.save(
        {
            "version": 1,
            "job_id": info["job_id"],
            "site_name": info["site_name"],
            "round": current_round,
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "loaders": [loader.state_dict() for loader in data_loaders],
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:]),
            "torch_rng": torch.get_rng_state(),
        },
        paths["output"],
    )
