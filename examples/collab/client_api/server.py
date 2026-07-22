# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Server-side FedAvg algorithm."""

import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.collab import collab


class FedAvg:
    """Server-side FedAvg algorithm.

    The server calls client.execute() which triggers client's receive().
    """

    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds

    @collab.main
    def run(self):
        """Execute the FedAvg algorithm."""
        print(f"Starting FedAvg for {self.num_rounds} rounds")
        print("=" * 60)

        global_weights = None

        for round_num in range(self.num_rounds):
            print(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

            # Create FLModel to send to clients
            input_model = FLModel(
                params=global_weights,
                current_round=round_num + 1,
                total_rounds=self.num_rounds,
            )

            # Call clients via execute() - triggers their receive()
            client_results = collab.clients.execute(
                fl_model=input_model,
                task_name="train",
                job_id="fedavg_job",
                site_name="",
            )

            # Aggregate results
            global_weights, global_loss = self._aggregate(client_results)

            print(f"  [Server] Global avg loss: {global_loss:.4f}")

        # Signal clients to stop
        print("\nSending stop signal to clients...")
        collab.clients.stop()

        print("\n" + "=" * 60)
        print(f"FedAvg completed after {self.num_rounds} rounds")

        return global_weights

    def _aggregate(self, client_results):
        """Aggregate client results using FedAvg."""
        valid_results = {}
        for client_id, error in client_results.failures.items():
            print(f"  Warning: {client_id} failed: {error}")
        for client_id, result in client_results:
            if result is None:
                continue
            valid_results[client_id] = result

        if not valid_results:
            raise RuntimeError("All clients failed!")

        all_weights = []
        all_losses = []
        for result in valid_results.values():
            if isinstance(result, FLModel):
                all_weights.append(result.params)
                all_losses.append(result.metrics.get("loss", 0.0))

        # Average model parameters
        if all_weights and all_weights[0]:
            avg_weights = {}
            for key in all_weights[0].keys():
                stacked = torch.stack([w[key].float() for w in all_weights])
                avg_weights[key] = stacked.mean(dim=0)
        else:
            avg_weights = None

        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
        return avg_weights, avg_loss
