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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Prepare TCGA patient splits and per-site bundles for the Collab job."""

import numpy as np
import torch
from breastgfcl import ParallelServerGFedCL
from federated.runtime import snapshot_rng
from federated.state import cpu_tree
from torch.utils.data import DataLoader, TensorDataset


def smoke_loaders(opt):
    """Small deterministic fixture; never used by the real TCGA run."""
    rng = np.random.default_rng(opt.seed)
    opt.client_spatial_features = [
        rng.normal(size=(opt.num_clients, 3)).astype(np.float32) for _ in range(opt.num_task)
    ]
    opt.client_temporal_features = [
        rng.normal(size=(opt.num_clients, 2)).astype(np.float32) for _ in range(opt.num_task)
    ]
    loaders = {}
    for client_id in range(opt.num_clients):
        loaders[client_id] = {}
        for task in range(opt.num_task):
            loaders[client_id][task] = {}
            for split in ("train", "test"):
                x = torch.from_numpy(rng.normal(size=(8, opt.input_dim)).astype(np.float32))
                y = torch.arange(8) % opt.num_classes
                loaders[client_id][task][split] = DataLoader(
                    TensorDataset(x, y),
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle and split == "train",
                )
    return loaders


def dataset_tensors(dataset):
    if not len(dataset):
        raise ValueError("Cannot package an empty client/task dataset")
    # Read by index to preserve the existing split/order; no second partition.
    samples = [dataset[index] for index in range(len(dataset))]
    return {
        "x": torch.stack([x for x, _ in samples]).cpu(),
        "y": torch.as_tensor([int(y) for _, y in samples], dtype=torch.long),
    }


def prepare_bundles(opt, smoke=False):
    loaders = smoke_loaders(opt) if smoke else None
    # Initialize models and data; NVFlare supplies the transport at execution time.
    workflow = ParallelServerGFedCL(opt, dataloaders=loaders)
    sites, client_states = [], []
    for client_id, client in enumerate(workflow.clients):
        data, metadata = {}, {}
        for task in range(opt.num_task):
            data[task] = {
                split: dataset_tensors(workflow.dataloaders[client_id][task][split].dataset)
                for split in ("train", "test")
            }
            labels = data[task]["train"]["y"]
            metadata[task] = {
                "label_counts": torch.bincount(labels, minlength=opt.num_classes),
                "batch_sizes": [
                    min(opt.batch_size, len(labels) - start) for start in range(0, len(labels), opt.batch_size)
                ],
            }
        site_opt = {
            key: value
            for key, value in vars(opt).items()
            if not key.startswith("client_spatial") and not key.startswith("client_temporal")
        }
        sites.append({"opt": cpu_tree(site_opt), "client_id": client_id, "data": data})
        client_states.append(
            {
                "weights": cpu_tree(client.get_weights()),
                "training_state": cpu_tree(client.get_training_state()),
                "task_metadata": metadata,
            }
        )
    server_bundle = {
        "opt": cpu_tree(vars(opt)),
        "discriminator": cpu_tree(workflow.server.get_discriminator()),
        "graph_state": cpu_tree(workflow.dygat.state_dict()),
        "clients": client_states,
        "rng_state": snapshot_rng(workflow.device),
    }
    return workflow, server_bundle, sites
