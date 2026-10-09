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

"""BreastG-FCL's multi-stage continual-learning workflow on Collab."""

import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from federated.runtime import restore_rng
from federated.state import capture_final_state, load_bundle, validate_device
from federated.transport import CollabTransport, ProxyClient

from nvflare.collab import collab

LOGGER = logging.getLogger(__name__)


class BreastGFCLServer:
    """Run the existing E/F/G/D coordinator with addressed Collab clients."""

    def __init__(self, bundle_name="server.pt"):
        self.bundle_name = bundle_name

    @collab.main
    def run(self):
        from breastgfcl import ParallelServerGFedCL
        from model.modules import BreastGraphGenerator
        from model.server import Server

        if collab.is_aborted:
            raise RuntimeError("NVFlare workflow aborted before initialization")
        bundle = load_bundle(self.bundle_name)
        opt = SimpleNamespace(**bundle["opt"])
        validate_device(opt)
        if len(bundle["clients"]) != opt.num_clients:
            raise ValueError("Prepared server bundle does not contain every client")
        expected_sites = {f"site-{index + 1}" for index in range(opt.num_clients)}
        actual_sites = [client.name for client in collab.clients]
        if len(actual_sites) != len(set(actual_sites)) or set(actual_sites) != expected_sites:
            raise RuntimeError(f"NVFlare roster mismatch: expected {sorted(expected_sites)}, got {actual_sites}")
        server = Server(opt)
        server.set_discriminator(bundle["discriminator"])
        graph_generator = BreastGraphGenerator(opt).to(opt.device)
        graph_generator.load_state_dict(bundle["graph_state"])
        transport = CollabTransport(opt)
        clients = [ProxyClient(i, opt, state, transport) for i, state in enumerate(bundle["clients"])]
        dummy_loaders = {
            i: {task: {"train": None, "test": None} for task in range(opt.num_task)} for i in range(opt.num_clients)
        }
        workflow = ParallelServerGFedCL.from_components(
            opt,
            server,
            graph_generator,
            clients,
            dummy_loaders,
            transport,
        )
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Initialization above must not change graph/privacy randomness.
        restore_rng(bundle["rng_state"])
        metrics, rounds, _quality = workflow.train_GFedCL()
        if collab.is_aborted:
            raise RuntimeError("NVFlare workflow aborted before final state export")
        final = capture_final_state(workflow, metrics, rounds)
        torch.save(final, output_dir / "final_state.pt")
        LOGGER.info("NVFlare shared workflow completed; saved %s", output_dir / "final_state.pt")
        return final
