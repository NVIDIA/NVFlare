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

"""Client script for MLP training on ESM2 embeddings."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier on protein embeddings")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--results-path", type=str, required=True, help="Path to inference results")
    parser.add_argument("--aggregation-epochs", type=int, default=4, help="Number of epochs per round")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--embedding-dimensions", type=int, default=320, help="ESM2 embedding dimensions")

    args = parser.parse_args()

    # Initialize NVFlare client
    flare.init()

    # (optional) metrics tracking
    summary_writer = SummaryWriter()

    # Get the site name for this client
    site_name = flare.get_site_name()
    print(f"\n[Site={site_name}] Training MLP classifier\n")

    # Check if we're simulating local training only
    sim_local = os.getenv("SIM_LOCAL", "False").lower() in ("true", "1", "yes")
    print(f"[Site={site_name}] Simulating local training only!")

    # Construct paths for this client
    data_path = os.path.join(args.data_root, f"data_{site_name}.csv")
    inference_result_path = os.path.join(args.results_path, f"inference_results_{site_name}", "predictions__rank_0.pt")

    # Read embeddings
    try:
        print(f"[Site={site_name}] Loading embeddings from {inference_result_path}")
        results = torch.load(inference_result_path, weights_only=False)
        protein_embeddings = results["embeddings"]
        print(f"[Site={site_name}] Loaded {len(protein_embeddings)} embeddings")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Inference results not found at {inference_result_path}. Did you run the inference job first?"
        )
    except KeyError:
        raise KeyError("Inference results file doesn't contain 'embeddings' key. File may be corrupted.")

    # Read labels
    try:
        print(f"[Site={site_name}] Loading labels from {data_path}")
        labels = pd.read_csv(data_path).astype(str)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Check that data preparation completed successfully."
        )

    # Prepare the data for training
    X_train, y_train = [], []
    X_test, y_test = [], []

    for index, label in labels.iterrows():
        embedding = protein_embeddings[index].to(torch.float32).numpy()
        if label["SET"] == "train":
            X_train.append(embedding)
            y_train.append(label["labels"])
        elif label["SET"] == "test":
            X_test.append(embedding)
            y_test.append(label["labels"])

    print(f"[Site={site_name}] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Define class labels
    class_labels = [
        "Cell_membrane",
        "Cytoplasm",
        "Endoplasmic_reticulum",
        "Extracellular",
        "Golgi_apparatus",
        "Lysosome",
        "Mitochondrion",
        "Nucleus",
        "Peroxisome",
        "Plastid",
    ]

    # Initialize MLP model
    model = MLPClassifier(
        solver="adam",
        hidden_layer_sizes=(512, 256, 128),
        batch_size=args.batch_size,
        learning_rate_init=args.lr,
        verbose=True,
    )

    # Initialize the model with random data (required by scikit-learn)
    _X, _y = [], []
    for lbl in class_labels:
        _X.append(np.random.rand(args.embedding_dimensions))
        _y.append(lbl)
    model.partial_fit(_X, _y, classes=class_labels)

    # Training loop with NVFlare
    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        print(f"\n[Site={site_name}] Round {input_model.current_round}: Received global model\n")

        # Load global weights (unless simulating local training)
        if not sim_local and input_model.params:
            coefs = []
            intercepts = []
            for name in sorted(input_model.params.keys()):
                if "coef" in name:
                    coefs.append(input_model.params[name])
                elif "intercept" in name:
                    intercepts.append(input_model.params[name])
                else:
                    raise ValueError(f"Expected name to contain either `coef` or `intercept` but it was `{name}`.")
            model.coefs_ = coefs
            model.intercepts_ = intercepts

        # Evaluate received global model
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        print(
            f"[Site={site_name}] Round {input_model.current_round}: Train accuracy: {train_accuracy * 100:.2f}%, Test accuracy: {test_accuracy * 100:.2f}%"
        )

        # Log metrics to tracking
        summary_writer.add_scalar(tag="train_accuracy", scalar=train_accuracy, global_step=input_model.current_round)
        summary_writer.add_scalar(tag="accuracy", scalar=test_accuracy, global_step=input_model.current_round)

        # Train the model
        for epoch in range(args.aggregation_epochs):
            model.partial_fit(X_train, y_train)

        # Get local weights
        local_weights = {}
        for i, w in enumerate(model.coefs_):
            local_weights[f"coef_{i}"] = w
        for i, w in enumerate(model.intercepts_):
            local_weights[f"intercept_{i}"] = w

        # Send full model weights back to server (simplified - no weight differences)
        output_model = flare.FLModel(
            params=local_weights,
            params_type=input_model.params_type,
            metrics={"accuracy": float(test_accuracy), "train_accuracy": float(train_accuracy)},
            meta={"NUM_STEPS_CURRENT_ROUND": args.aggregation_epochs * len(X_train)},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
