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

"""Addressed Collab calls preserving the BreastG-FCL coordinator interface."""

import torch
from federated.runtime import operation_seed
from federated.state import cpu_tree

from nvflare.collab import collab


class ProxyClient:
    """Server-side client state; raw datasets remain at the corresponding site."""

    def __init__(self, client_id, opt, bundle, transport):
        self.client_id = int(client_id)
        self.opt = opt
        self.device = torch.device(opt.device)
        self.transport = transport
        self._weights = cpu_tree(bundle["weights"])
        self._training_state = cpu_tree(bundle["training_state"])
        self._task_metadata = {int(key): cpu_tree(value) for key, value in bundle["task_metadata"].items()}
        self._discriminator_state = None

    def getId(self):
        return self.client_id

    def register_task(self, task, dataloader):
        del dataloader
        counts_by_task = self._training_state.setdefault("task_label_counts", {})
        if task in counts_by_task:
            return
        metadata = self._task_metadata[task]
        counts = metadata["label_counts"].detach().cpu().long().clone()
        sizes = list(metadata["batch_sizes"])
        if (
            tuple(counts.shape) != (self.opt.num_classes,)
            or (counts < 0).any()
            or not sizes
            or any(not isinstance(size, int) or size < 1 for size in sizes)
            or int(counts.sum()) != sum(sizes)
        ):
            raise ValueError(f"Invalid prepared replay metadata for client {self.client_id}, task {task}")
        counts_by_task[task] = counts
        self._training_state.setdefault("task_batch_sizes", {})[task] = sizes

    def set_server_discriminator(self, state):
        self._discriminator_state = cpu_tree(state)

    def get_weights(self):
        return cpu_tree(self._weights)

    def set_weights(self, weights):
        for key in ("encoder", "predictor", "generator"):
            if key in weights:
                self._weights[key] = cpu_tree(weights[key])

    def get_training_state(self):
        return cpu_tree(self._training_state)

    def set_training_state(self, state):
        self._training_state = cpu_tree(state)

    def test(self, task_id, dataloader, relational_graphs):
        # Final evaluation calls client.test directly in the shared coordinator.
        pending = self.transport.test_client(self, task_id, dataloader, relational_graphs)
        return self.transport.gather([pending])[0]


class CollabTransport:
    """Call each site's published methods with only that site's model state."""

    def __init__(self, opt):
        self.opt = opt
        self._round_index = 0
        self._task = 0
        self._timeout = int(getattr(opt, "nvflare_timeout", 300))
        if self._timeout < 1:
            raise ValueError("nvflare_timeout must be a positive integer")

    def set_round(self, task, round_index):
        self._task = int(task)
        self._round_index = int(round_index)

    def _submit(self, operation, client, task, graphs, dataloader, epochs=1):
        del dataloader
        if collab.is_aborted:
            raise RuntimeError("NVFlare workflow aborted")
        client_id = client.getId()
        if not isinstance(client_id, int) or not 0 <= client_id < self.opt.num_clients:
            raise ValueError(f"Invalid client identity: {client_id}")
        site = f"site-{client_id + 1}"
        remote = collab.get_clients([site])(blocking=False, timeout=self._timeout)
        stream = getattr(remote, operation)(
            task=int(task),
            graphs=cpu_tree(graphs),
            weights=client.get_weights(),
            training_state=client.get_training_state(),
            discriminator=cpu_tree(client._discriminator_state),
            seed=operation_seed(self.opt.seed, task, self._round_index, client_id, operation),
            epochs=int(epochs),
        )
        return site, stream

    def generate_encodings(self, client, task, graphs, loader):
        return self._submit("encode", client, task, graphs, loader)

    def train_client(self, client, task, graphs, loader, epochs):
        return self._submit("train", client, task, graphs, loader, epochs)

    def test_client(self, client, task, loader, graphs):
        return self._submit("test", client, task, graphs, loader)

    def gather(self, pending):
        """Collect complete site outcomes in the coordinator's client order."""
        results = []
        for site, stream in pending:
            if collab.is_aborted:
                raise RuntimeError("NVFlare workflow aborted while waiting for clients")
            # Nonblocking calls expose failures only after their stream has
            # completed. Never treat an empty or partial round as successful.
            outcomes = list(stream)
            if collab.is_aborted:
                raise RuntimeError("NVFlare workflow aborted while waiting for clients")
            if stream.failures:
                raise next(iter(stream.failures.values()))
            if len(outcomes) != 1 or outcomes[0][0] != site:
                raise RuntimeError(f"Expected exactly one Collab result from {site}")
            results.append(outcomes[0][1])
        return results
