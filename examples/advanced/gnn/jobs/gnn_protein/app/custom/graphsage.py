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

import os.path

import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

# (1) import nvflare client API
import nvflare.client as flare

# (optional) set a fix place to help check the dataset on each client
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "PPI")
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"


def main():
    # Define local epochs
    epochs = 60

    # Create PPI dataset for training.
    train_dataset = PPI(DATASET_PATH, split="train")
    val_dataset = PPI(DATASET_PATH, split="val")
    test_dataset = PPI(DATASET_PATH, split="test")

    # Group all training graphs into a single graph to perform sampling:
    train_data = Batch.from_data_list(train_dataset)
    loader = LinkNeighborLoader(
        train_data,
        batch_size=2048,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
        num_workers=6,
        persistent_workers=True,
    )
    print("finish setup train loader")

    # Evaluation loaders (one datapoint corresponds to a graph)
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)
    print("finish setup eval loaders")

    # Model
    model = GraphSAGE(
        in_channels=train_dataset.num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=64,
    )

    # (2) initializes NVFlare client API
    flare.init()

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (3) loads model from NVFlare
        model.load_state_dict(input_model.params)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        # (optional) use GPU to speed things up
        model.to(DEVICE)
        # (optional) calculate total steps
        steps = epochs * len(loader)
        for epoch in range(epochs):
            # model.train()
            running_loss = instance_count = 0
            for data in enumerate(loader, 0):
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

            print(f"Epoch: {epoch:02d}, Loss: {running_loss/instance_count:.4f}")

        print("Finished Training")

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            model = GraphSAGE(
                in_channels=train_dataset.num_features,
                hidden_channels=64,
                num_layers=2,
                out_channels=64,
            )
            model.load_state_dict(input_weights)
            # (optional) use GPU to speed things up
            model.to(DEVICE)

            def encode(data_loader):
                model.to(device)
                model.eval()

                xs, ys = [], []
                for data in data_loader:
                    data = data.to(device)
                    xs.append(model(data.x, data.edge_index).cpu())
                    ys.append(data.y.cpu())
                return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

            # Train classifier on training set:
            with torch.no_grad():
                x, y = encode(train_loader)

            clf = MultiOutputClassifier(SGDClassifier(loss="log_loss", penalty="l2"))
            clf.fit(x, y)

            train_f1 = f1_score(y, clf.predict(x), average="micro")

            # Evaluate on validation set:
            x, y = encode(val_loader)
            val_f1 = f1_score(y, clf.predict(x), average="micro")

            # Evaluate on test set:
            x, y = encode(test_loader)
            test_f1 = f1_score(y, clf.predict(x), average="micro")

            return train_f1, val_f1, test_f1

        # (6) evaluate on received model for model selection
        _, _, global_test_f1 = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"test_f1": global_test_f1},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
