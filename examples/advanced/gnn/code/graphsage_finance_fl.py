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
from process_elliptic import process_ellipitc
from pyg_sage import SAGE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

DEVICE = "cuda:0"

# (1) import nvflare client API
import nvflare.client as flare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/datasets/elliptic_pp",
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
        default=0,
        help="0: use all data, 1-N: use data from client N",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
    )
    args = parser.parse_args()

    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, str(args.client_id)))

    # Create elliptic dataset for training.
    df_classes = pd.read_csv(os.path.join(args.data_path, "txs_classes.csv"))
    df_edges = pd.read_csv(os.path.join(args.data_path, "txs_edgelist.csv"))
    df_features = pd.read_csv(os.path.join(args.data_path, "txs_features.csv"))

    # Preprocess data
    node_features, classified_idx, unclassified_idx, edge_index, weights, labels, y_train = process_ellipitc(
        df_features, df_edges, df_classes
    )

    # Converting data to PyG graph data format
    train_data = Data(
        x=node_features, edge_index=edge_index, edge_attr=weights, y=torch.tensor(labels, dtype=torch.double)
    )

    # Splitting data into train and validation
    _, _, y_train, _, train_idx, valid_idx = train_test_split(
        node_features[classified_idx], y_train, classified_idx, test_size=0.1, random_state=77, stratify=y_train
    )
    # Futher split train data into two clients
    _, _, _, _, train_1_idx, train_2_idx = train_test_split(
        node_features[train_idx], y_train, train_idx, test_size=0.5, random_state=77, stratify=y_train
    )

    # Get the subgraph index for the client
    # note that client 0 uses all data
    # client 1 uses data from classified_1 and unclassified data
    # client 2 uses data from classified_2 and unclassified data
    if args.client_id == 0:
        train_data_sub = train_data
    if args.client_id == 1:
        train_data_sub = train_data.subgraph(torch.tensor(train_1_idx.append(unclassified_idx)))
        train_idx = np.arange(len(train_1_idx))
    elif args.client_id == 2:
        train_data_sub = train_data.subgraph(torch.tensor(train_2_idx.append(unclassified_idx)))
        train_idx = np.arange(len(train_2_idx))
    train_data = train_data.to(DEVICE)
    train_data_sub = train_data_sub.to(DEVICE)

    # Train model
    model = SAGE(train_data_sub.num_node_features, hidden_channels=256, num_classes=2, num_layers=3)
    model.double()

    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (3) loads model from NVFlare
        model.load_state_dict(input_model.params)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # (optional) use GPU to speed things up
        model.to(DEVICE)
        # (optional) calculate total steps
        steps = args.epochs * len(train_idx)
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            # perform full batch training using all classified data
            out = model(train_data_sub)
            loss = F.nll_loss(out[train_idx], train_data_sub.y[train_idx].T.to(torch.long))
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
            writer.add_scalar("train_loss", loss.item(), input_model.current_round * args.epochs + epoch)

            # (5) wraps evaluation logic into a method to re-use for
            #       evaluation on both trained and received model
            def evaluate(input_weights):
                model_eval = SAGE(train_data.num_node_features, hidden_channels=256, num_classes=2, num_layers=3)
                model_eval.double()
                model_eval.load_state_dict(input_weights)
                # (optional) use GPU to speed things up
                model_eval.to(DEVICE)

                with torch.no_grad():
                    model_eval.eval()
                    out = model_eval(train_data)
                    y_pred = torch.argmax(out, dim=1).detach().cpu().numpy()
                    y_true = train_data.y.detach().cpu().numpy()
                    val_auc = roc_auc_score(y_true[valid_idx], y_pred[valid_idx])
                    print(f"Validation AUC: {val_auc:.4f} ")
                    writer.add_scalar("val_auc", val_auc, input_model.current_round * args.epochs + epoch)
                return val_auc

        # (6) evaluate on received model for model selection
        global_auc = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"validation_auc": global_auc},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
