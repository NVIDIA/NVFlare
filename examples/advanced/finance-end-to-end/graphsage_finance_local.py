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
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

DEVICE = "cuda:0"


def edge_index_gen(df_feat_class, df_edges):
    # Sort the data by UETR
    df_feat_class = df_feat_class.sort_values(by="UETR").reset_index(drop=True)

    # Generate UETR-index map with the feature list
    node_ids = df_feat_class["UETR"].values
    map_id = {j: i for i, j in enumerate(node_ids)}  # mapping nodes to indexes

    # Get class labels
    labels = df_feat_class["Class"].values

    # Map UETR to indexes in the edge map
    edges = df_edges.copy()
    edges.UETR_1 = edges.UETR_1.map(map_id)
    edges.UETR_2 = edges.UETR_2.map(map_id)
    edges = edges.astype(int)

    # for undirected graph
    edge_index = np.array(edges.values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.float)

    # UETR mapped to corresponding indexes, drop UETR and class
    node_features = df_feat_class.drop(["UETR", "Class"], axis=1).copy()
    node_features = torch.tensor(np.array(node_features.values), dtype=torch.float)

    return node_features, edge_index, weights, node_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/dataset/horizontal_credit_fraud_data/ZHSZUS33_Bank_1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/nvflare/gnn/finance_local",
    )
    args = parser.parse_args()

    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_path))

    # Create elliptic dataset for training.
    df_data = pd.read_csv(os.path.join(args.data_path, "train_normalized.csv"), index_col=0)
    # Drop irrelevant columns
    df_feat_class = df_data.drop(
        columns=["Currency_Country", "Beneficiary_BIC", "Currency", "Receiver_BIC", "Sender_BIC"]
    )
    # Use GNN in unsupervised mode, drop the class column
    df_edges = pd.read_csv(os.path.join(args.data_path, "train_edgemap.csv"), header=None)
    # Add column names to the edge map
    df_edges.columns = ["UETR_1", "UETR_2"]

    # Preprocess data
    node_features, edge_index, weights, ids, labels = edge_index_gen(df_feat_class, df_edges)

    # Converting data to PyG graph data format
    train_data = Data(x=node_features, edge_index=edge_index, edge_attr=weights)

    # Define the dataloader for graphsage training
    loader = LinkNeighborLoader(
        train_data,
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
    )

    # Model
    model = GraphSAGE(
        in_channels=node_features.shape[1],
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = instance_count = 0
        for data in loader:
            # get the inputs data
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

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.output_path, "model.pt"))

    # Load the model and perform inference / encoding
    model_enc = GraphSAGE(
        in_channels=node_features.shape[1],
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )
    model_enc.load_state_dict(torch.load(os.path.join(args.output_path, "model.pt")))
    model_enc.eval()
    # Perform encoding
    h = model_enc(node_features, edge_index)
    embed = pd.DataFrame(h.cpu().detach().numpy())
    # Add column names as V_0, V_1, ... V_63
    embed.columns = [f"V_{i}" for i in range(embed.shape[1])]
    # Concatenate the node ids and class labels with the encoded features
    embed["UETR"] = ids
    embed["Class"] = labels
    # Move the UETR and Class columns to the front
    embed = embed[["UETR", "Class"] + [col for col in embed.columns if col not in ["UETR", "Class"]]]
    embed.to_csv(os.path.join(args.output_path, "encoded_features.csv"), index=False)


if __name__ == "__main__":
    main()
