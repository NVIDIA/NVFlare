# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Evaluate an NVFLARE SimpleCNN checkpoint on the clean CIFAR-10 test set."""

import argparse

import torch
from src.model import SimpleCNN


def evaluate(checkpoint_path, test_path, batch_size, device):
    """Evaluate a checkpoint on the clean test set."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = SimpleCNN()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    test_data = torch.load(
        test_path,
        map_location="cpu",
        weights_only=True,
    )

    images = test_data["images"]
    labels = test_data["labels"]

    if len(images) == 0:
        raise ValueError("Evaluation dataset is empty")

    if len(labels) == 0:
        raise ValueError("Evaluation labels are empty")

    if len(images) != len(labels):
        raise ValueError(f"Evaluation images and labels have different lengths: " f"{len(images)} vs {len(labels)}")

    correct = 0
    total = len(labels)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_images = images[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            outputs = model(batch_images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == batch_labels).sum().item()

    if total == 0:
        raise ValueError("Evaluation dataset contains zero samples")

    accuracy = 100.0 * correct / total

    return accuracy, correct, total


def main():
    """Run checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a SimpleCNN checkpoint on CIFAR-10.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the NVFLARE checkpoint.",
    )
    parser.add_argument(
        "--test_data",
        default="research/fedscs/data/test.pt",
        help="Path to the clean test dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device.",
    )

    args = parser.parse_args()

    accuracy, correct, total = evaluate(
        args.checkpoint,
        args.test_data,
        args.batch_size,
        args.device,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
