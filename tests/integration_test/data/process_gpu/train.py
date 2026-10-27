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

"""Small deterministic real CPU/CUDA regression, shared by both lifetimes."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import nvflare.client as flare

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
parser.add_argument("--gpu-oom", action="store_true")
parser.add_argument("--cpu-site", choices=("site-1", "site-2"))
args = parser.parse_args()
torch.set_num_threads(1)
flare.init()
device = "cpu" if flare.get_site_name() == args.cpu_site else args.device
expected_gpus = 1 if device == "cuda" else 0
if torch.cuda.device_count() != expected_gpus:
    raise RuntimeError(f"expected {expected_gpus} visible GPUs; observed {torch.cuda.device_count()}")
if expected_gpus and not torch.cuda.is_available():
    raise RuntimeError("CUDA qualification requires real CUDA kernels")

while flare.is_running():
    received = flare.receive()
    if received is None:
        break
    model = torch.nn.Linear(2, 1).to(device)
    model.load_state_dict({key: torch.as_tensor(value) for key, value in received.params.items()})
    inputs = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=device)
    targets = inputs @ torch.tensor([[2.0], [-3.0]], device=device)
    initial_loss = torch.nn.functional.mse_loss(model(inputs), targets).item()
    if flare.is_train():
        if args.gpu_oom and flare.get_site_name() == "site-1":
            print(json.dumps({"event": "process_gpu_oom_probe", "pid": os.getpid(), "ppid": os.getppid()}), flush=True)
            torch.empty(torch.cuda.get_device_properties(0).total_memory + 1, dtype=torch.uint8, device="cuda")
            raise AssertionError("oversized CUDA allocation unexpectedly succeeded")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
    final_loss = torch.nn.functional.mse_loss(model(inputs), targets).item()
    if device == "cuda":
        torch.cuda.synchronize()
    if not final_loss < 1e-4:
        raise RuntimeError(f"model did not converge: {final_loss}")
    params = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    output_dir = Path("process-gpu-models")
    output_dir.mkdir(exist_ok=True)
    torch.save(params, output_dir / f"{os.getpid()}-{received.current_round}.pt")
    facts = {
        "event": "process_gpu_compute",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "site": flare.get_site_name(),
        "round": received.current_round,
        "task": "train" if flare.is_train() else "validate",
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": torch.cuda.device_count(),
        "mse_before": initial_loss,
        "mse": final_loss,
        "gpu_name": torch.cuda.get_device_name(0) if expected_gpus else None,
        "allocated_bytes": torch.cuda.max_memory_allocated() if expected_gpus else 0,
        "finished_at": time.time(),
    }
    (output_dir / f"{os.getpid()}-{received.current_round}.json").write_text(json.dumps(facts, sort_keys=True))
    print(json.dumps(facts, sort_keys=True), flush=True)
    # Model selection scores the received global model, before local updates.
    metrics = {"mse": initial_loss, "accuracy": max(0.0, 100.0 * (1.0 - initial_loss))}
    if flare.is_train():
        flare.send(flare.FLModel(params=params, metrics=metrics, meta={"NUM_STEPS_CURRENT_ROUND": 100}))
    else:
        flare.send(flare.FLModel(metrics=metrics))
