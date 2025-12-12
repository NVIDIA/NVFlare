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
"""
client side training scripts
"""

import argparse
import os

import numpy as np
import torch
from model import Network
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


class EmbeddingDataset(Dataset):
    """Dataset for loading embeddings and targets from .npy files"""

    def __init__(self, embeddings_path, targets_path):
        """
        Args:
            embeddings_path: Path to the embeddings .npy file
            targets_path: Path to the targets .npy file
        """
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.targets = np.load(targets_path).astype(np.float32)

        print(f"Loaded embeddings: {self.embeddings.shape}")
        print(f"Loaded targets: {self.targets.shape}")

        if len(self.embeddings) != len(self.targets):
            raise ValueError(f"Mismatch between embeddings ({len(self.embeddings)}) and targets ({len(self.targets)})")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.targets[idx])


def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # Initialize model
    model = Network()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use MSE loss for regression
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    total_rounds = flare.receive().total_rounds  # get total rounds from NVFlare

    # Cosine annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * total_rounds, eta_min=0.01 * lr)
    print(f"Using CosineAnnealingLR with T_max={epochs * total_rounds}, eta_min={0.01 * lr}")

    # Load embeddings and targets from federated folder
    embeddings_path = os.path.join(args.data_prefix, client_name, "train_data.embeddings.npy")
    targets_path = os.path.join(args.data_prefix, client_name, "train_data.targets.npy")

    print("Loading data from:")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Targets: {targets_path}")

    train_dataset = EmbeddingDataset(embeddings_path, targets_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(train_dataset)} samples for {client_name}")
    assert len(train_dataset) > 0, "No training data found"

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                embeddings, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(embeddings)
                cost = loss_fn(predictions, targets)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy()

            avg_loss = running_loss / len(train_loader)
            print(
                f"site={client_name}, Epoch: {epoch}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            global_step = input_model.current_round * epochs + epoch
            summary_writer.add_scalar(tag="loss_for_each_epoch", scalar=float(avg_loss), global_step=global_step)
            summary_writer.add_scalar(tag="learning_rate", scalar=scheduler.get_last_lr()[0], global_step=global_step)
            # Step the learning rate scheduler after each epoch
            scheduler.step()

        print(f"Finished Training for {client_name}")

        PATH = f"./{client_name}_model.pth"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_prefix", type=str, default="/data/federated_data")
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--lr", type=float, default=0.001)
    args = args.parse_args()

    main(args)
