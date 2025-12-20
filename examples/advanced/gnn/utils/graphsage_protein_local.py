# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

np.random.seed(77)
DEVICE = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/datasets/ppi",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--total_clients",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--client_id",
        type=int,
        default=2,
        help="0: use all data, 1-N: use data from client N",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/nvflare/gnn/protein_local",
    )
    args = parser.parse_args()

    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, str(args.client_id)))

    # Create PPI dataset for training.
    train_dataset = PPI(args.data_path, split="train")
    val_dataset = PPI(args.data_path, split="val")

    # Group all training graphs into a single graph to perform sampling:
    train_data = Batch.from_data_list(train_dataset)

    # Split the training graph into subgraphs according to the number of clients
    node_idx = np.arange(train_data.num_nodes)
    np.random.shuffle(node_idx)
    client_idx = np.split(node_idx, args.total_clients)

    # Get the subgraph for the client
    if args.client_id == 0:
        train_data_sub = train_data
    else:
        subset_idx = torch.tensor(client_idx[args.client_id - 1])
        train_data_sub = train_data.subgraph(subset_idx)

    # Define the dataloader for graphsage training
    loader = LinkNeighborLoader(
        train_data_sub,
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
    )

    # Evaluation loaders (one datapoint corresponds to a graph)
    # As the main training is unsupervised, additional model needs to be trained for classification
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Model
    model = GraphSAGE(
        in_channels=train_dataset.num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model.to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = instance_count = 0
        for data in tqdm.tqdm(loader):
            # get the inputs data
            # (optional) use GPU to speed things up
            data = data.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            h = model(data.x, data.edge_index)
            h_src = h[data.edge_label_index[0]]
            h_dst = h[data.edge_label_index[1]]
            link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
            loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
            loss.backward()
            optimizer.step()
            # add record
            running_loss += float(loss.item()) * link_pred.numel()
            instance_count += link_pred.numel()
        print(f"Epoch: {epoch:02d}, Loss: {running_loss / instance_count:.4f}")
        writer.add_scalar("train_loss", running_loss / instance_count, epoch)

        if epoch % 10 == 0:
            # Evaluate model
            def encode(data_loader):
                model.eval()
                xs, ys = [], []
                for data in data_loader:
                    data = data.to(DEVICE)
                    xs.append(model(data.x, data.edge_index).cpu())
                    ys.append(data.y.cpu())
                return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

            with torch.no_grad():
                # Train classifier on training set:
                x, y = encode(train_loader)
                clf = MultiOutputClassifier(SGDClassifier(loss="log_loss", penalty="l2"))
                clf.fit(x, y)
                # Evaluate on validation set:
                x, y = encode(val_loader)
                val_f1 = f1_score(y, clf.predict(x), average="micro")
                print(f"Validation F1: {val_f1:.4f}")
                writer.add_scalar("validation_f1", val_f1, epoch)


if __name__ == "__main__":
    main()
