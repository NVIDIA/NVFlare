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

"""Backend qualification probe; no training or recipe implementation changes."""

import argparse
import json
import os

import torch

import nvflare.client as flare

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
parser.add_argument("--gpu-count", type=int, default=0)
parser.add_argument("--allocation-mb", type=int, default=64)
args = parser.parse_args()
torch.set_num_threads(1)
if torch.cuda.device_count() != args.gpu_count:
    raise RuntimeError(f"expected {args.gpu_count} GPUs, observed {torch.cuda.device_count()}")
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA qualification requires actual CUDA computation")

flare.init()
while flare.is_running():
    model = flare.receive()
    if model is None:
        break
    devices = [f"cuda:{index}" for index in range(args.gpu_count)] if args.device == "cuda" else ["cpu"]
    checksum, peak_bytes = 0.0, 0
    for device in devices:
        # A real allocation and kernels; visibility/nvidia-smi alone cannot pass.
        tensor = torch.ones(args.allocation_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        tensor.square_().add_(1)
        checksum += tensor.sum().item()
        if args.device == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes += torch.cuda.max_memory_allocated(device)
        del tensor
    print(
        json.dumps(
            {
                "event": "docker_task_compute",
                "pid": os.getpid(),
                "device": args.device,
                "gpu_uuids": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
                "gpu_count": args.gpu_count,
                "checksum": checksum,
                "peak_bytes": peak_bytes,
                "round": model.current_round,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    flare.send(
        flare.FLModel(
            params=model.params,
            metrics={
                "gpu_count": args.gpu_count,
                "checksum": checksum,
                "peak_bytes": peak_bytes,
            },
        )
    )
