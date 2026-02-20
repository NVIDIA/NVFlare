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

"""B1 pass-through integration test client script.

Mirrors the llm_hf/client.py pattern (launch_once=True external process,
while-loop FL rounds) but uses a simple synthetic-data MLP so no dataset
download or GPU is required.

The LargeNet model (~8 MB of float32 parameters) exceeds the 2 MB streaming
threshold, which forces ViaDownloaderDecomposer to route tensors through the
download service.  With B1 PASS_THROUGH enabled in ClientAPILauncherExecutor,
the CJ creates LazyDownloadRef placeholders instead of materialising tensors;
this subprocess then downloads each tensor directly from the FL server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from large_model_net import LargeNet

# (1) import nvflare client API
import nvflare.client as flare

DEVICE = "cpu"  # CPU-only: no GPU required for integration testing
BATCH_SIZE = 4
INPUT_DIM = 1024
NUM_CLASSES = 10


def main():
    net = LargeNet()
    net.to(DEVICE)

    # (2) initialise NVFlare client API
    flare.init()

    # (3) FL training loop — mirrors llm_hf/client.py structure
    while flare.is_running():
        # (4) receive global model from NVFlare (triggers B1 pass-through download)
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}", flush=True)

        # (5) load global model weights into the local network
        net.load_state_dict(input_model.params)

        # (6) one step of simulated training on synthetic data (no dataset needed)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()

        print(f"round {input_model.current_round} training loss: {loss.item():.4f}", flush=True)

        # (7) construct and send trained model back to NVFlare
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": 0.5},  # placeholder metric
            meta={"NUM_STEPS_CURRENT_ROUND": 1},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
