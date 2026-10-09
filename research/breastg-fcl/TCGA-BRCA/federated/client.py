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

"""Site-local BreastG-FCL training exposed through the Collab API."""

import threading
from types import SimpleNamespace

from federated.runtime import execute_client, seeded_rng
from federated.state import cpu_tree, load_bundle, validate_device
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import collab


class _AbortAwareLoader:
    """Preserve loader behavior while checking Collab cancellation per batch."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if collab.is_aborted:
                raise RuntimeError("NVFlare client operation aborted")
            yield batch

    def __getattr__(self, name):
        return getattr(self.loader, name)


class BreastGFCLClient:
    """Keep one site's dataset, E/F/G models, and optimizer/task state."""

    def __init__(self, bundle_name="site.pt"):
        self.bundle_name = bundle_name
        self._client = None
        self._dataloaders = None
        self._client_id = None
        self._lock = threading.Lock()

    @collab.init
    def initialize(self):
        from model.client import ModifiedClient

        # Model construction consumes RNG too. Preserve the process RNG when
        # simulator sites share a process, just as for each training operation.
        with self._lock, seeded_rng():
            if collab.is_aborted:
                raise RuntimeError("NVFlare client initialization aborted")
            bundle = load_bundle(self.bundle_name)
            opt = SimpleNamespace(**bundle["opt"])
            validate_device(opt)
            client_id = int(bundle["client_id"])
            if not 0 <= client_id < opt.num_clients:
                raise ValueError("Prepared site identity is outside the configured client roster")
            expected_site = f"site-{client_id + 1}"
            if collab.site_name != expected_site:
                raise ValueError(f"Prepared bundle belongs to {expected_site}, not {collab.site_name}")
            loaders = {}
            for task, splits in bundle["data"].items():
                task = int(task)
                loaders[task] = {}
                for split in ("train", "test"):
                    values = splits[split]
                    x, y = values["x"], values["y"]
                    if len(x) != len(y):
                        raise ValueError(f"Site dataset sample/label mismatch for task {task}, {split}")
                    loaders[task][split] = DataLoader(
                        TensorDataset(x, y),
                        batch_size=opt.batch_size,
                        shuffle=bool(opt.shuffle) if split == "train" else False,
                        num_workers=getattr(opt, "num_workers", 0),
                        pin_memory=getattr(opt, "pin_memory", False),
                    )
            if set(loaders) != set(range(opt.num_task)):
                raise ValueError("Prepared site dataset does not contain every configured task")
            self._client = ModifiedClient(client_id, opt)
            self._dataloaders = loaders
            self._client_id = client_id

    def _execute(self, operation, task, graphs, weights, training_state, discriminator, seed, epochs):
        with self._lock, seeded_rng():
            if collab.is_aborted:
                raise RuntimeError("NVFlare client operation aborted")
            if self._client is None:
                raise RuntimeError("BreastG-FCL client is not initialized")
            if collab.site_name != f"site-{self._client_id + 1}":
                raise ValueError("NVFlare client identity changed during the run")
            if operation not in ("encode", "train", "test") or task not in self._dataloaders:
                raise ValueError("Invalid client operation or task")
            client = self._client
            client.set_weights(weights)
            client.set_training_state(training_state)
            if discriminator is not None:
                client.set_server_discriminator(discriminator)
            elif operation == "train":
                raise ValueError("Client training requires server discriminator state")
            loader = _AbortAwareLoader(self._dataloaders[task]["test" if operation == "test" else "train"])
            result = execute_client(operation, client, task, graphs, loader, epochs=epochs, seed=seed)
            if collab.is_aborted:
                raise RuntimeError("NVFlare client operation aborted")
            return cpu_tree(result)

    @collab.publish
    def encode(self, task, graphs, weights, training_state, discriminator, seed, epochs=1):
        return self._execute("encode", task, graphs, weights, training_state, discriminator, seed, epochs)

    @collab.publish
    def train(self, task, graphs, weights, training_state, discriminator, seed, epochs=1):
        return self._execute("train", task, graphs, weights, training_state, discriminator, seed, epochs)

    @collab.publish
    def test(self, task, graphs, weights, training_state, discriminator, seed, epochs=1):
        return self._execute("test", task, graphs, weights, training_state, discriminator, seed, epochs)
