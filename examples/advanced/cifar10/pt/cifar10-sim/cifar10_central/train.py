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

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10_data_utils import create_data_loaders, create_datasets
from model import ModerateCNN
from torch.utils.tensorboard import SummaryWriter
from train_utils import evaluate, get_lr_values

# Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable cuDNN auto-tuner for optimal algorithms (speeds up training with fixed input sizes)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def main(args):
    model = ModerateCNN()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Initialize learning rate scheduler
    scheduler = None
    if not args.no_lr_scheduler:
        eta_min = args.lr * args.cosine_lr_eta_min_factor  # Minimum LR is a factor of the initial LR
        T_max = args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"Using CosineAnnealingLR scheduler: initial_lr={args.lr}, eta_min={eta_min}, T_max={T_max}")

    print("Creating datasets for centralized training")
    train_dataset, valid_dataset = create_datasets("central", train_idx_root=args.train_idx_root, central=True)
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Initialize TensorBoard writer
    summary_writer = SummaryWriter(log_dir=args.output_dir)
    print(f"TensorBoard logs will be saved to: {args.output_dir}")

    # Move model to GPU
    model.to(DEVICE)

    # Evaluate initial model
    val_acc = evaluate(model, valid_loader)
    print(f"Initial model accuracy on validation set: {100 * val_acc:.2f} %")
    summary_writer.add_scalar("val_acc", val_acc, 0)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    curr_lr = get_lr_values(optimizer)[0]  # Initialize learning rate before training loop

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Report loss at the end of each epoch
        avg_loss = running_loss / len(train_loader)
        curr_lr = get_lr_values(optimizer)[0]  # Get current learning rate
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Average Loss: {avg_loss:.4f} - Learning Rate: {curr_lr:.6f}")

        # Log metrics to TensorBoard
        summary_writer.add_scalar("train_loss", avg_loss, epoch)
        summary_writer.add_scalar("learning_rate", curr_lr, epoch)

        # Evaluate the model on validation set
        val_acc = evaluate(model, valid_loader)
        print(f"Model accuracy on validation set: {100 * val_acc:.2f} %\n")
        summary_writer.add_scalar("val_acc", val_acc, epoch)

        # Step the learning rate scheduler at the end of each epoch
        if scheduler is not None:
            scheduler.step()

    print("Finished training!")

    # Close TensorBoard writer
    summary_writer.close()

    # Save the final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_idx_root",
        type=str,
        default="/tmp/cifar10_splits",
        help="Root directory for training index. Default is /tmp/cifar10_splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs. Default is 4.",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate. Default is 1e-2.")
    parser.add_argument(
        "--no_lr_scheduler",
        action="store_true",
        help="Whether to disable the learning rate scheduler. Default is False.",
    )
    parser.add_argument(
        "--cosine_lr_eta_min_factor",
        type=float,
        default=0.01,
        help="Minimum learning rate as a factor of the initial learning rate for cosine annealing scheduler. Default is 0.01 (or 1%).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading. Default is 2.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/nvflare/simulation/cifar10_central",
        help="Directory to save the trained model. Default is /tmp/nvflare/simulation/cifar10_central.",
    )
    args = parser.parse_args()

    main(args)
