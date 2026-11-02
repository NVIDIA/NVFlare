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

import json
import logging
from pathlib import Path

import numpy as np

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import log_progress


class FedAvg:
    def __init__(self, num_rounds=3):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        weights = np.zeros(3)
        history = []
        for round_index in range(self.num_rounds):
            # A blocking group call fans out to all clients and returns (site, result) pairs.
            responses = collab.clients(timeout=60).train(weights)
            results = dict(responses)
            if responses.failures or not results:
                raise RuntimeError(f"Client training failed: {responses.failures}")
            total = sum(result["samples"] for result in results.values())
            weights = sum(result["weights"] * result["samples"] for result in results.values()) / total
            loss = sum(result["loss"] * result["samples"] for result in results.values()) / total
            history.append({"round": round_index + 1, "local_loss": loss})
            log_progress(
                logging.getLogger(__name__), f"Round {round_index + 1}/{self.num_rounds}: local loss={loss:.4f}"
            )
        output = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        np.save(output / "weights.npy", weights)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        return weights
