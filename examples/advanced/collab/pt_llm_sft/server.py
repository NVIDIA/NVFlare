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

"""Synchronous server workflow and aggregation for federated SFT."""

from pathlib import Path

import torch

from nvflare.collab import collab


def average_model_states(updates: dict[str, dict], min_clients: int) -> tuple[dict[str, torch.Tensor], float]:
    """Apply sample-weighted FedAvg to complete PyTorch model states."""
    if len(updates) < min_clients:
        raise RuntimeError(f"received {len(updates)} successful client updates; need at least {min_clients}")

    first_update = next(iter(updates.values()))
    expected_keys = set(first_update["weights"])
    total_examples = sum(update["num_examples"] for update in updates.values())
    if total_examples <= 0:
        raise RuntimeError("client updates contain no training examples")

    for site_name, update in updates.items():
        if set(update["weights"]) != expected_keys:
            raise ValueError(f"{site_name} returned a different set of model parameters")
        if update["num_examples"] <= 0:
            raise ValueError(f"{site_name} returned an invalid example count")

    averaged = {}
    for name in sorted(expected_keys):
        reference = first_update["weights"][name]
        for site_name, update in updates.items():
            if update["weights"][name].shape != reference.shape:
                raise ValueError(f"{site_name} returned an incompatible shape for {name}")

        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, device="cpu", dtype=torch.float32)
            for update in updates.values():
                tensor = update["weights"][name].detach().to(device="cpu", dtype=torch.float32)
                accumulator.add_(tensor, alpha=update["num_examples"])
            averaged[name] = (accumulator / total_examples).to(dtype=reference.dtype)
        else:
            averaged[name] = reference.detach().cpu().clone()

    average_loss = sum(update["train_loss"] * update["num_examples"] for update in updates.values()) / total_examples
    return averaged, average_loss


class SFTFedAvg:
    """Coordinate synchronous client training and sample-weighted aggregation."""

    def __init__(
        self,
        num_epochs: int,
        syncs_per_epoch: int,
        min_clients: int,
        output_root: str,
        call_timeout: float,
        save_every_sync: bool,
    ):
        self.num_epochs = num_epochs
        self.syncs_per_epoch = syncs_per_epoch
        self.min_clients = min_clients
        self.output_root = output_root
        self.call_timeout = call_timeout
        self.save_every_sync = save_every_sync

    # @collab.main marks the single server-side entry point that drives the workflow.
    @collab.main
    def run(self) -> dict[str, torch.Tensor]:
        output_dir = Path(self.output_root) / "server"
        output_dir.mkdir(parents=True, exist_ok=True)

        # This proxy call invokes the client's published get_model_state method on client 0.
        global_weights = collab.clients[0](timeout=self.call_timeout).get_model_state()

        total_syncs = self.num_epochs * self.syncs_per_epoch
        for sync_number in range(1, total_syncs + 1):
            epoch_number = (sync_number - 1) // self.syncs_per_epoch + 1
            sync_in_epoch = (sync_number - 1) % self.syncs_per_epoch + 1
            print(f"=== Epoch {epoch_number}/{self.num_epochs}, " f"sync {sync_in_epoch}/{self.syncs_per_epoch} ===")

            # .train matches @collab.publish on LLMSFTClient, calls all clients, and records failures separately.
            call_results = collab.clients(timeout=self.call_timeout).train(sync_number, global_weights)
            updates = dict(call_results)
            for site_name, error in call_results.failures.items():
                print(f"Warning: {site_name} failed: {error}")

            # The returned example counts weight each successful client's full model update.
            global_weights, average_loss = average_model_states(updates, self.min_clients)
            message = f"Aggregated {len(updates)} clients: train_loss={average_loss:.4f}"
            if self.save_every_sync:
                checkpoint = output_dir / f"model_sync_{sync_number}.pt"
                torch.save(global_weights, checkpoint)
                message += f"; saved {checkpoint}"
            print(message)

        final_checkpoint = output_dir / "model_final.pt"
        torch.save(global_weights, final_checkpoint)
        print(f"Training complete. Final model: {final_checkpoint}")
        return global_weights
