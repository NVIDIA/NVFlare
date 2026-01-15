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
Client-side training script with Differential Privacy (DP-SGD) using Opacus
"""

import argparse

import torch
import torch.nn as nn
from model import TabularMLP
from opacus import PrivacyEngine
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def load_data(client_id, n_clients, batch_size):
    """Load and partition Credit Card Fraud dataset"""
    print("Loading Credit Card Fraud Detection dataset...")

    # Load dataset from OpenML
    data = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    X = data.data.values
    y = data.target.values.astype(int)  # Convert to binary: 0=normal, 1=fraud

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution - Normal: {(y == 0).sum()}, Fraud: {(y == 1).sum()}")

    # Split data across clients
    client_data_size = len(X) // n_clients
    start_idx = client_id * client_data_size
    end_idx = (client_id + 1) * client_data_size if client_id < n_clients - 1 else len(X)

    X_client = X[start_idx:end_idx]
    y_client = y[start_idx:end_idx]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(model, data_loader, device):
    """Evaluate model using Accuracy and F1 Score"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics using sklearn
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    return accuracy, f1, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=1.0,
        help="Target epsilon for differential privacy (lower = more private)",
    )
    parser.add_argument("--target_delta", type=float, default=1e-5, help="Target delta for differential privacy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--n_clients", type=int, default=2, help="Total number of clients")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.001

    # Model definition - dimensions will be determined from incoming model
    # Placeholder model just to initialize (will be overwritten by server model)
    model = TabularMLP(input_dim=29, hidden_dims=[64, 32], output_dim=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # (2) Initialize NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # Extract client ID from site name (e.g., "site-1" -> 0)
    client_id = int(client_name.split("-")[-1]) - 1

    # Load data for this client
    train_loader, test_loader = load_data(client_id, args.n_clients, batch_size)

    # Get actual feature dimensions from data
    sample_batch = next(iter(train_loader))
    n_features = sample_batch[0].shape[1]
    n_classes = 2  # Binary classification: normal vs fraud

    print(f"Data loaded: {n_features} features, {n_classes} classes")

    # (optional) metrics tracking
    summary_writer = SummaryWriter()

    print(f"Client {client_name}: Using Differential Privacy with epsilon={args.target_epsilon}")
    print("Note: Privacy budget will accumulate across ALL federated rounds")
    print(f"Model: {n_features} features -> 64 -> 32 -> {n_classes} classes (fraud detection)")

    # Initialize privacy engine once (outside the training loop)
    privacy_engine = None

    while flare.is_running():
        # (3) Receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        # (4) Loads model from NVFlare
        # Handle Opacus-wrapped model state dict
        if privacy_engine is not None and hasattr(model, "_module"):
            # Opacus wraps the model, so we need to add "_module." prefix
            global_params = {}
            for k, v in input_model.params.items():
                global_params[f"_module.{k}"] = v
            model.load_state_dict(global_params, strict=True)
        else:
            model.load_state_dict(input_model.params, strict=True)
        model.to(device)

        # Evaluate the global model
        global_test_accuracy, global_test_f1, global_test_precision, global_test_recall = evaluate(
            model, test_loader, device
        )
        summary_writer.add_scalar(
            tag="global_test_accuracy", scalar=global_test_accuracy, global_step=input_model.current_round
        )
        summary_writer.add_scalar(tag="global_test_f1", scalar=global_test_f1, global_step=input_model.current_round)
        summary_writer.add_scalar(
            tag="global_test_precision", scalar=global_test_precision, global_step=input_model.current_round
        )
        summary_writer.add_scalar(
            tag="global_test_recall", scalar=global_test_recall, global_step=input_model.current_round
        )

        print(
            f"site={client_name}, Global Test Accuracy: {global_test_accuracy:.4f}, Global Test F1: {global_test_f1:.4f}"
        )

        # Train the model with differential privacy
        model.train()

        # ====== DIFFERENTIAL PRIVACY SETUP (First Round Only) ======
        # Initialize privacy engine ONLY in first round to track budget across ALL rounds
        if input_model.current_round == 0:
            print("\n=== Initializing Differential Privacy (Round 0) ===")
            print(f"Target epsilon: {args.target_epsilon}")
            print(f"Target delta: {args.target_delta}")
            print(f"Max gradient norm: {args.max_grad_norm}")

            # Calculate total epochs across ALL federated rounds for privacy accounting
            total_epochs_all_rounds = epochs * input_model.total_rounds

            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=total_epochs_all_rounds,  # Total across ALL rounds
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                max_grad_norm=args.max_grad_norm,
            )

            # Get the automatically computed noise multiplier
            noise_multiplier = optimizer.noise_multiplier
            print(f"Computed noise multiplier: {noise_multiplier:.4f}")
            print(f"Privacy budget will accumulate across {input_model.total_rounds} rounds")
            print(f"Total epochs for privacy accounting: {total_epochs_all_rounds}")
            print("=" * 60)
        # ========================================

        steps = epochs * len(train_loader)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print every epoch
            avg_loss = running_loss / len(train_loader)
            summary_writer.add_scalar(
                tag="train_loss", scalar=avg_loss, global_step=input_model.current_round * steps + epoch
            )
            print(f"site={client_name}, Epoch: {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            running_loss = 0.0

        # Evaluate on train and test sets at the end of the round
        print(f"\nEvaluating model after round {input_model.current_round}...")
        train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader, device)
        test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, device)

        print(
            f"Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}"
        )
        print(
            f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}"
        )

        # Log metrics to TensorBoard
        summary_writer.add_scalar(tag="train_accuracy", scalar=train_accuracy, global_step=input_model.current_round)
        summary_writer.add_scalar(tag="test_accuracy", scalar=test_accuracy, global_step=input_model.current_round)
        summary_writer.add_scalar(tag="train_f1_score", scalar=train_f1, global_step=input_model.current_round)
        summary_writer.add_scalar(tag="test_f1_score", scalar=test_f1, global_step=input_model.current_round)

        print(
            f"Round {input_model.current_round} - Train: Acc={train_accuracy:.4f}, F1={train_f1:.4f} | Test: Acc={test_accuracy:.4f}, F1={test_f1:.4f}"
        )

        # Print cumulative privacy budget spent
        epsilon = privacy_engine.get_epsilon(args.target_delta)
        print(f"\n=== Privacy Budget (Round {input_model.current_round}/{input_model.total_rounds - 1}) ===")
        print(f"Cumulative ε = {epsilon:.2f} (target: {args.target_epsilon})")
        print(f"δ = {args.target_delta}")
        if input_model.current_round < input_model.total_rounds - 1:
            print("⚠️  Privacy budget is accumulating! More rounds remaining.")
        else:
            print("✓ Final cumulative privacy budget after all rounds")
        print("=" * 60)

        summary_writer.add_scalar(tag="privacy_epsilon", scalar=epsilon, global_step=input_model.current_round)

        print(f"Finished Training for {client_name}")

        # (6) Construct trained FL model
        # Extract original model weights (remove "_module." prefix from Opacus)
        model_state_dict = model.cpu().state_dict()
        clean_params = {}
        for k, v in model_state_dict.items():
            clean_key = k.replace("_module.", "")
            clean_params[clean_key] = v

        output_model = flare.FLModel(
            params=clean_params,
            metrics={
                "accuracy": global_test_accuracy,  # Global model accuracy (for server model selection)
                "f1_score": global_test_f1,
                "precision": global_test_precision,
                "recall": global_test_recall,
                "privacy_epsilon": epsilon,
            },
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")

        # (7) Send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
