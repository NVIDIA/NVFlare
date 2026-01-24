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
Script to generate a pre-trained model for evaluation.
This trains a model on CIFAR-10 and saves it to a checkpoint file.
"""

import argparse

import torch
from model import CIFAR10DataModule, LitNet
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="pretrained_model.pt", help="Output checkpoint path")

    return parser.parse_args()


def main():
    args = define_parser()

    print("=" * 80)
    print("Generating Pre-trained Model")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Output checkpoint: {args.output}")
    print("=" * 80)

    # Create model and data module
    model = LitNet()
    cifar10_dm = CIFAR10DataModule(batch_size=args.batch_size)

    # Setup trainer
    if torch.cuda.is_available():
        trainer = Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1)
    else:
        trainer = Trainer(max_epochs=args.epochs, devices=None)

    # Train the model
    print("\nStarting training...")
    trainer.fit(model, datamodule=cifar10_dm)

    # Validate to get final accuracy
    print("\nValidating trained model...")
    trainer.validate(model, datamodule=cifar10_dm)

    # Save the full LitNet state dict (includes model + metrics)
    print(f"\nSaving model to {args.output}...")
    torch.save(model.state_dict(), args.output)

    print("\n" + "=" * 80)
    print(f"✓ Pre-trained model saved to: {args.output}")
    print("  You can now run job.py to evaluate this model across clients.")
    print("=" * 80)


if __name__ == "__main__":
    main()
