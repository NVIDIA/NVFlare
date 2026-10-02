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

"""State copying and prepared local bundles shared by Collab components."""

import copy
from pathlib import Path

import numpy as np
import torch

from nvflare.collab import collab


def cpu_tree(value):
    """Detach transport/checkpoint tensors without pickle-based RPC decoding."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def capture_final_state(workflow, metrics, rounds):
    """Snapshot all trained state and metrics without retaining live tensors."""
    return cpu_tree(
        {
            "weights": workflow.clients[0].get_weights(),
            "discriminator": workflow.server.get_discriminator(),
            "discriminator_optimizer": workflow.server.optimizer_D.state_dict(),
            "discriminator_scheduler": workflow.server.lr_scheduler_D.state_dict(),
            "graph_state": workflow.dygat.state_dict(),
            "training_states": [client.get_training_state() for client in workflow.clients],
            "metrics": metrics,
            "rounds": rounds,
        }
    )


def load_bundle(bundle_name):
    """Load a prepared artifact from this site's application data directory."""
    if not isinstance(bundle_name, str) or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a filename within app/config/data")
    job_id = collab.fl_ctx.get_job_id()
    if job_id is None:
        raise RuntimeError("NVFlare job identity is missing")
    path = Path(collab.workspace.get_app_dir(job_id)) / "config" / "data" / bundle_name
    # Bundles are locally prepared job artifacts, never untrusted RPC bytes.
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_device(opt):
    device = torch.device(opt.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The prepared experiment requires CUDA, but this site has no CUDA device")
    return device
