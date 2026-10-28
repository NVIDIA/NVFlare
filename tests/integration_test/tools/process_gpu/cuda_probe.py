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

"""Shared real-CUDA probe for integration tests and qualification tools.

Importing this helper does not run CUDA or require pytest. Launch only on an
explicitly approved device assignment.
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

from nvflare.apis.task_launcher_spec import TaskLaunchRequest, TaskResourceRequest

COMPUTE = """
import json, os, subprocess, sys, time
from pathlib import Path
import torch
output, mode = Path(sys.argv[1]), sys.argv[2]
torch.set_num_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
model = torch.nn.Linear(1, 1, bias=False).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
x, y = torch.ones(64, 1, device='cuda'), torch.full((64, 1), 3., device='cuda')
for _ in range(40):
    optimizer.zero_grad()
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
assert loss.item() < 1e-4
torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output.with_suffix('.pt'))
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']) if mode == 'child' else None
record = json.dumps(dict(pid=os.getpid(), descendant_pid=child.pid if child else None,
    visible=os.environ['CUDA_VISIBLE_DEVICES'], loss=loss.item(),
    allocated_bytes=torch.cuda.max_memory_allocated(), computed_at=time.time()))
pending = output.with_suffix('.pending')
pending.write_text(record)
pending.replace(output)
if mode in ('hold', 'child'):
    time.sleep(120)
if mode == 'exception':
    raise RuntimeError('deliberate post-CUDA worker exception')
if mode == 'abrupt':
    os._exit(17)
if mode == 'oom':
    try:
        torch.empty(torch.cuda.get_device_properties(0).total_memory + 1, dtype=torch.uint8, device='cuda')
    except torch.cuda.OutOfMemoryError as error:
        output.with_suffix('.failure.json').write_text(json.dumps(dict(error_type=type(error).__name__, message=str(error))))
        raise
    raise AssertionError('oversized CUDA allocation unexpectedly succeeded')
"""


def launch_probe(launcher, output, mode="normal"):
    return launcher.launch_task(
        TaskLaunchRequest(
            "gpu-qualification",
            "site-probe",
            "compute",
            str(uuid.uuid4()),
            (sys.executable, "-c", COMPUTE, str(output), mode),
            environment={"PATH": os.environ["PATH"]},
            resources=TaskResourceRequest(gpu_count=1),
        )
    )


def wait_for_probe(path, timeout=60):
    """Wait for a complete record, including while an older writer is mid-write."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"GPU probe did not publish a complete JSON record: {path}") from None
            time.sleep(min(0.05, remaining))
