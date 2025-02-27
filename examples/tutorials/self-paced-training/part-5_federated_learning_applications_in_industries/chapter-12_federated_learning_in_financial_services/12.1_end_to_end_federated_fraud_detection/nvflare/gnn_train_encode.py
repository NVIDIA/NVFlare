# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# (1) import nvflare client API
import nvflare.client as flare


def edge_index_gen(df_feat_class, df_edges):
    # Sort the data by UETR
    df_feat_class = df_feat_class.sort_values(by="UETR").reset_index(drop=True)

    # Generate UETR-index map with the feature list
    node_id = df_feat_class["UETR"].values
    map_id = {j: i for i, j in enumerate(node_id)}  # mapping nodes to indexes

    # Get class labels
    label = df_feat_class["Class"].values

    # Map UETR to indexes in the edge map
    edges = df_edges.copy()
    edges.UETR_1 = edges.UETR_1.map(map_id)
    edges.UETR_2 = edges.UETR_2.map(map_id)
    edges = edges.astype(int)

    # for undirected graph
    edge_index = np.array(edges.values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weight = torch.tensor([1] * edge_index.shape[1], dtype=torch.float)

    # UETR mapped to corresponding indexes, drop UETR and class
    node_feat = df_feat_class.drop(["UETR", "Class"], axis=1).copy()
    node_feat = torch.tensor(np.array(node_feat.values), dtype=torch.float)

    return node_feat, edge_index, weight, node_id, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
        default="/tmp/dataset/credit_data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="/tmp/dataset/credit_data",
    )
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()
    site_name = flare.get_site_name()

    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, site_name))

    # Load the data
    dataset_names = ["train", "test"]

    node_features = {}
    edge_indices = {}
    weights = {}
    node_ids = {}
    labels = {}

    for ds_name in dataset_names:
        # Get feature and class
        file_name = os.path.join(args.data_path, site_name, f"{ds_name}_normalized.csv")
        df = pd.read_csv(file_name, index_col=0)
        # Drop irrelevant columns
        df = df.drop(columns=["Currency_Country", "Beneficiary_BIC", "Currency", "Receiver_BIC", "Sender_BIC"])
        df_feat_class = df
        # Get edge map
        file_name = os.path.join(args.data_path, site_name, f"{ds_name}_edgemap.csv")
        df = pd.read_csv(file_name, header=None)
        # Add column names to the edge map
        df.columns = ["UETR_1", "UETR_2"]
        df_edges = df

        # Preprocess data
        node_feat, edge_index, weight, node_id, label = edge_index_gen(df_feat_class, df_edges)
        node_features[ds_name] = node_feat
        edge_indices[ds_name] = edge_index
        weights[ds_name] = weight
        node_ids[ds_name] = node_id
        labels[ds_name] = label

    # Converting training data to PyG graph data format
    train_data = Data(x=node_features["train"], edge_index=edge_indices["train"], edge_attr=weights["train"])

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
        in_channels=node_features["train"].shape[1],
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}/{input_model.total_rounds}")

        # (4) loads model from NVFlare
        model.load_state_dict(input_model.params)

        # (5) perform encoding for both training and test data
        def gnn_encode(model_param, node_feature, edge_index, id, label):
            # Load the model and perform inference / encoding
            model_enc = GraphSAGE(
                in_channels=node_feature.shape[1],
                hidden_channels=64,
                num_layers=2,
                out_channels=64,
            )
            model_enc.load_state_dict(model_param)
            model_enc.to(DEVICE)
            model_enc.eval()
            node_feature = node_feature.to(DEVICE)
            edge_index = edge_index.to(DEVICE)

            # Perform encoding
            h = model_enc(node_feature, edge_index)
            embed = pd.DataFrame(h.cpu().detach().numpy())
            # Add column names as V_0, V_1, ... V_63
            embed.columns = [f"V_{i}" for i in range(embed.shape[1])]
            # Concatenate the node ids and class labels with the encoded features
            embed["UETR"] = id
            embed["Class"] = label
            # Move the UETR and Class columns to the front
            embed = embed[["UETR", "Class"] + [col for col in embed.columns if col not in ["UETR", "Class"]]]
            return embed

        # Only do encoding for the last round
        if input_model.current_round == input_model.total_rounds - 1:
            print("Encoding the data with the final model")
            for ds_name in dataset_names:
                embed = gnn_encode(
                    input_model.params,
                    node_features[ds_name],
                    edge_indices[ds_name],
                    node_ids[ds_name],
                    labels[ds_name],
                )
                embed.to_csv(os.path.join(args.output_path, site_name, f"{ds_name}_embedding.csv"), index=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.to(DEVICE)
        steps = args.epochs * len(loader)
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
            writer.add_scalar(
                "train_loss", running_loss / instance_count, input_model.current_round * args.epochs + epoch
            )

        print("Finished Training")
        # Save the model
        torch.save(model.state_dict(), os.path.join(args.output_path, site_name, "model.pt"))

        # (6) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"loss": running_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
