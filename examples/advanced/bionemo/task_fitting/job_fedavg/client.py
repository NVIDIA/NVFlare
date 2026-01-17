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
"""
PyTorch-based federated learning client for MLP training on protein embeddings
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import CLASS_LABELS, ProteinMLP
from torch.utils.data import DataLoader, TensorDataset

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def evaluate_model(model, data_loader, criterion, device, compute_loss=True):
    """
    Evaluate model on given data loader.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader containing evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        compute_loss: Whether to compute and return loss (default: True)

    Returns:
        tuple: (accuracy, avg_loss) if compute_loss=True, else (accuracy, None)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if compute_loss:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader) if compute_loss else None

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Federated MLP Training Client")
    parser.add_argument("--data-root", type=str, required=True, help="Path to data root directory")
    parser.add_argument("--results-path", type=str, required=True, help="Path to inference results")
    parser.add_argument("--aggregation-epochs", type=int, default=4, help="Number of local training epochs per round")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--embedding-dimensions", type=int, default=1280, help="Embedding dimensions")
    args = parser.parse_args()

    # Initialize NVFlare client API
    flare.init()
    site_name = flare.get_site_name()
    print(f"[Site={site_name}] Starting MLP training client")

    # Initialize summary writer for tracking metrics
    summary_writer = SummaryWriter()

    # Check if running in local training mode
    sim_local = os.environ.get("SIM_LOCAL", "False").lower() == "true"
    if sim_local:
        print(f"[Site={site_name}] Running in LOCAL training mode (ignoring global model)")
    else:
        print(f"[Site={site_name}] Running in FEDERATED training mode")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Site={site_name}] Using device: {device}")

    # Construct paths
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
        labels_df = pd.read_csv(data_path).astype(str)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Check that data preparation completed successfully."
        )

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_LABELS)}

    # Prepare the data for training
    X_train, y_train = [], []
    X_test, y_test = [], []

    for index, row in labels_df.iterrows():
        embedding = protein_embeddings[index].to(torch.float32)
        label_idx = label_to_idx[row["labels"]]

        if row["SET"] == "train":
            X_train.append(embedding)
            y_train.append(label_idx)
        elif row["SET"] == "test":
            X_test.append(embedding)
            y_test.append(label_idx)

    # Convert to tensors
    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.stack(X_test)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"[Site={site_name}] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = ProteinMLP(input_dim=args.embedding_dimensions, num_classes=len(CLASS_LABELS))
    model = model.to(device)
    print(f"[Site={site_name}] Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with NVFlare
    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        print(f"\n[Site={site_name}] Round {input_model.current_round}: Received global model")

        # Load global weights (unless simulating local training)
        if not sim_local:
            model.load_state_dict(input_model.params, strict=True)
            print(f"[Site={site_name}] Loaded global model weights")
        model.to(device)

        # Evaluate received global model
        train_accuracy, _ = evaluate_model(model, train_loader, criterion, device, compute_loss=False)
        test_accuracy, _ = evaluate_model(model, test_loader, criterion, device, compute_loss=False)

        # Log metrics to summary writer
        global_step = input_model.current_round
        summary_writer.add_scalar(tag="train_accuracy", scalar=train_accuracy, global_step=global_step)
        summary_writer.add_scalar(tag="accuracy", scalar=test_accuracy, global_step=global_step)

        print(
            f"[Site={site_name}] Global Model - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        # Local training
        model.train()
        for epoch in range(args.aggregation_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()

            epoch_accuracy = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            print(
                f"[Site={site_name}] Epoch {epoch + 1}/{args.aggregation_epochs} - Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
            )

        # Evaluate after training
        final_train_accuracy, _ = evaluate_model(model, train_loader, criterion, device, compute_loss=False)
        final_test_accuracy, avg_test_loss = evaluate_model(model, test_loader, criterion, device, compute_loss=True)

        print(
            f"[Site={site_name}] After training - Train Accuracy: {final_train_accuracy:.4f}, Test Accuracy: {final_test_accuracy:.4f}"
        )

        # Prepare metrics
        metrics = {
            "accuracy": test_accuracy,  # global model test accuracy used for global model selection
            "train_accuracy": train_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "final_train_accuracy": final_train_accuracy,
            "loss": avg_test_loss,
        }

        # Send updated model back to server
        output_model = flare.FLModel(
            params=model.state_dict(),
            metrics=metrics,
            meta={"num_steps": len(train_loader) * args.aggregation_epochs},
        )
        flare.send(output_model)
        print(f"[Site={site_name}] Sent updated model to server\n")


if __name__ == "__main__":
    main()
