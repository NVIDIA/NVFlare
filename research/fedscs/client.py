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

"""CIFAR-10 client for the FedSCS NVIDIA FLARE example."""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN
from torch.utils.data import DataLoader, TensorDataset

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_client_dataset(data_dir, site_name):
    """Load the prepared dataset for one client."""
    client_id = site_name.split("-")[-1]

    client_file = os.path.join(
        data_dir,
        "clients",
        f"client_{client_id}.pt",
    )

    if not os.path.isfile(client_file):
        raise FileNotFoundError(f"Client dataset not found: {client_file}")

    data = torch.load(
        client_file,
        weights_only=True,
    )

    if "images" not in data or "labels" not in data:
        raise ValueError(f"Client dataset is missing 'images' or 'labels': {client_file}")

    if len(data["images"]) == 0 or len(data["labels"]) == 0:
        raise ValueError(f"Client dataset is empty: {client_file}")

    if len(data["images"]) != len(data["labels"]):
        raise ValueError(f"Client dataset has mismatched image/label counts: {client_file}")

    return TensorDataset(
        data["images"],
        data["labels"],
    )


def load_test_dataset(data_dir):
    """Load the common clean CIFAR-10 test dataset."""
    test_file = os.path.join(
        data_dir,
        "test.pt",
    )

    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test dataset not found: {test_file}")

    data = torch.load(
        test_file,
        weights_only=True,
    )

    if "images" not in data or "labels" not in data:
        raise ValueError(f"Test dataset is missing 'images' or 'labels': {test_file}")

    if len(data["images"]) == 0 or len(data["labels"]) == 0:
        raise ValueError(f"Test dataset is empty: {test_file}")

    if len(data["images"]) != len(data["labels"]):
        raise ValueError(f"Test dataset has mismatched image/label counts: {test_file}")

    return TensorDataset(
        data["images"],
        data["labels"],
    )


def evaluate(model, data_loader, criterion):
    """Evaluate a model and return loss and accuracy."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    if total == 0:
        raise ValueError("Evaluation data loader is empty. Check the dataset preparation.")

    return total_loss / total, correct / total


def compute_model_diff(local_model, global_model):
    """Compute local model minus received global model."""
    local_state = local_model.state_dict()
    global_state = global_model.state_dict()

    if set(local_state) != set(global_state):
        missing = sorted(set(global_state) - set(local_state))
        extra = sorted(set(local_state) - set(global_state))

        raise ValueError(
            "Local and global models have different parameter schemas. " f"Missing: {missing}; Extra: {extra}."
        )

    model_diff = {}

    for name in local_state:
        local_value = local_state[name].detach().cpu()
        global_value = global_state[name].detach().cpu()

        if local_value.shape != global_value.shape:
            raise ValueError(
                f"Parameter '{name}' has local shape "
                f"{tuple(local_value.shape)} but global shape "
                f"{tuple(global_value.shape)}."
            )

        model_diff[name] = local_value - global_value

    return model_diff


def compute_update_norm(model_diff):
    """Compute the L2 norm of a model DIFF update."""
    squared_norm = torch.tensor(
        0.0,
        dtype=torch.float64,
    )

    for name, value in model_diff.items():
        tensor = value.detach().to(dtype=torch.float64)

        if not torch.isfinite(tensor).all():
            raise ValueError(f"Model DIFF parameter '{name}' contains non-finite values.")

        squared_norm += torch.sum(tensor * tensor)

    if not torch.isfinite(squared_norm):
        raise ValueError("Model DIFF squared norm is non-finite.")

    update_norm = torch.sqrt(squared_norm)

    if not torch.isfinite(update_norm):
        raise ValueError("Model DIFF norm is non-finite.")

    return update_norm


def train_one_round(
    model,
    train_loader,
    optimizer,
    criterion,
    local_epochs,
):
    """Train the local model for one FL round."""
    model.train()

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Training data loader is empty. Check the dataset preparation.")

    return total_loss / total_batches


def main(args):
    """Run the NVFLARE client."""
    data_dir = os.path.abspath(args.data_dir)

    flare.init()

    site_name = flare.get_site_name()

    print("=" * 60)
    print(f"Starting FedSCS client: {site_name}")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    train_dataset = load_client_dataset(
        data_dir,
        site_name,
    )

    test_dataset = load_test_dataset(
        data_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"{site_name}: {len(train_dataset)} training samples")
    print(f"{site_name}: {len(test_dataset)} test samples")

    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
    )

    while flare.is_running():
        # ---------------------------------------------------------------
        # Receive the current global model.
        # ---------------------------------------------------------------

        input_model = flare.receive()

        if input_model is None:
            raise RuntimeError(f"{site_name}: Received no model from the server.")

        if input_model.params is None:
            raise RuntimeError(f"{site_name}: Received model has no parameters.")

        current_round = input_model.current_round

        print("")
        print("=" * 60)
        print(f"{site_name}: FL Round {current_round}")
        print("=" * 60)

        # ---------------------------------------------------------------
        # Load global model.
        # ---------------------------------------------------------------

        model.load_state_dict(
            input_model.params,
            strict=True,
        )

        # Keep an immutable copy of the received global model.
        global_model = copy.deepcopy(model)

        global_model.eval()

        for parameter in global_model.parameters():
            parameter.requires_grad = False

        # ---------------------------------------------------------------
        # Evaluate received global model on the clean test set.
        # ---------------------------------------------------------------

        global_loss, global_accuracy = evaluate(
            global_model,
            test_loader,
            criterion,
        )

        print(f"{site_name}: " f"Global test loss = {global_loss:.4f}, " f"accuracy = {100.0 * global_accuracy:.2f}%")

        # ---------------------------------------------------------------
        # Local training.
        # ---------------------------------------------------------------

        average_loss = train_one_round(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            local_epochs=args.local_epochs,
        )

        print(f"{site_name}: " f"Average local loss = {average_loss:.4f}")

        # ---------------------------------------------------------------
        # Evaluate locally trained model.
        # ---------------------------------------------------------------

        local_loss, local_accuracy = evaluate(
            model,
            test_loader,
            criterion,
        )

        print(f"{site_name}: " f"Local test loss = {local_loss:.4f}, " f"accuracy = {100.0 * local_accuracy:.2f}%")

        # ---------------------------------------------------------------
        # Compute DIFF:
        #
        # Delta w_i = w_i(local) - w(global)
        # ---------------------------------------------------------------

        model_diff = compute_model_diff(
            model,
            global_model,
        )

        diff_norm = compute_update_norm(model_diff)

        print(f"{site_name}: " f"Update norm = {diff_norm.item():.6f}")

        # ---------------------------------------------------------------
        # Send DIFF update to NVFLARE.
        # ---------------------------------------------------------------

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={
                "accuracy": local_accuracy,
                "loss": local_loss,
                "global_accuracy": global_accuracy,
                "global_loss": global_loss,
                "train_loss": average_loss,
                "diff_norm": diff_norm.item(),
            },
            meta={
                "client_name": site_name,
                "NUM_STEPS_CURRENT_ROUND": (args.local_epochs * len(train_loader)),
            },
        )

        flare.send(output_model)

        print(f"{site_name}: " f"Finished round {current_round}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=4,
        help="Number of local training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Local SGD learning rate.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
        ),
        help="Directory containing the prepared CIFAR-10 datasets.",
    )

    args = parser.parse_args()

    main(args)
