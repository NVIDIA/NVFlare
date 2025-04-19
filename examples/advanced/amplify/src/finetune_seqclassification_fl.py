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
Finetune AMPLIFY model for sequence regression.
"""

import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from model import AmplifyRegressor, print_model_info
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, DataCollatorWithPadding

# (1) import nvflare client API
import nvflare.client as flare


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AMPLIFY model for sequence regression")
    # Data paths
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",  # nvflare will define the working directory
        help="Directory to save model and logs",
    )
    # Pretrained model
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="chandar-lab/AMPLIFY_120M",
        help="Name or path of the pretrained AMPLIFY model",
    )
    # Hyper-parameters
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--trunk_lr", type=float, default=1e-4, help="Learning rate for the AMPLIFY trunk")
    parser.add_argument("--regressor_lr", type=float, default=1e-2, help="Learning rate for the regression layers")
    # Model architecture
    parser.add_argument(
        "--layer_sizes",
        type=str,
        default="128,64,32",
        help="Comma-separated list of layer sizes for the regression MLP",
    )
    # Training options
    parser.add_argument(
        "--frozen_trunk", action="store_true", help="Whether to freeze the AMPLIFY trunk during training"
    )
    # Deterministic training
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic training")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for deterministic training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seed for deterministic training
    set_seed(args.seed)

    # (2) initializes NVFlare client API
    flare.init()
    client_name = flare.get_site_name()

    # Start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up output directories
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, f"run_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    # Parse layer sizes from string to list of integers
    layer_sizes = [int(size) for size in args.layer_sizes.split(",")]

    # Build Regression model on top of AMPLIFY
    model = AmplifyRegressor(
        pretrained_model_name_or_path=args.pretrained_model, layer_sizes=layer_sizes
    )  # one output label for regression task

    # Print model architecture and configuration
    print_model_info(model, layer_sizes, args)

    # Load AMPLIFY tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)

    # Add data files
    data_files = {"train": args.train_csv, "test": args.test_csv}
    dataset = load_dataset("csv", data_files=data_files)

    # Set tokenizer
    dataset.set_transform(
        lambda x: {"labels": x["fitness"]}
        | tokenizer(x["combined"], padding=True, pad_to_multiple_of=8, return_tensors="pt")
    )

    # Create the dataloaders with deterministic worker initialization
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )

    # Build the loss, optimizer, and scheduler
    loss_fn = torch.nn.MSELoss()

    # Create parameter groups with different learning rates
    param_groups = [
        {"params": model.trunk.parameters(), "lr": args.trunk_lr},
        {"params": model.regressor.parameters(), "lr": args.regressor_lr},
    ]

    # Create single optimizer with parameter groups
    optimizer = torch.optim.AdamW(param_groups)

    # Create scheduler
    total_rounds = flare.receive().total_rounds  # get total rounds from NVFlare
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0,
        # lr schedule is applied after the first round
        total_iters=len(dataloader_train) * (args.n_epochs) * (total_rounds - 1),
    )

    # Main federated learning training loop
    global_step = 0
    while flare.is_running():
        # (3) receive FLModel from NVFlare
        input_model = flare.receive()

        print(f"current_round={input_model.current_round}, total_rounds={total_rounds}, input_model={input_model}")

        # (4) load global model from NVFlare
        missing_keys, unexpected_keys = model.load_state_dict(input_model.params, strict=False)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")
        model = model.to(device)

        # (5) Evaluate the global model on the test set
        global_mean_test_loss, global_rmse_test_loss = evaluate(model, dataloader_test, loss_fn, device)
        # Log test loss to TensorBoard
        writer.add_scalar(f"{client_name}/Loss/global_test", global_mean_test_loss, global_step)
        writer.add_scalar(f"{client_name}/RMSE/global_test", global_rmse_test_loss, global_step)

        # Training loop
        for epoch in range(args.n_epochs):
            train_loss = []
            model.train()
            for i, batch in enumerate(dataloader_train):
                # Convert to correct dtype and move to GPU
                input_ids = batch["input_ids"].to(torch.long).to(device)
                attention_mask = batch["attention_mask"].to(torch.float32).to(device)
                labels = batch["labels"].to(device)

                # Convert the attention mask to an additive mask
                attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))

                # Compute the model output
                output = model(input_ids, attention_mask, frozen_trunk=args.frozen_trunk)

                # Compute the loss and accuracy
                loss = loss_fn(output.squeeze(), labels)

                # Update the parameters
                loss.backward()
                optimizer.step()

                # Decay the learning rate only after the first round
                if input_model.current_round > 0:
                    scheduler.step()
                optimizer.zero_grad()

                # Log the loss and training progress
                train_loss.append(loss.item())
                current_loss = np.mean(train_loss)
                print(
                    f"\rEpoch: {epoch:6d}/{args.n_epochs - 1} Step {i:6d}/{len(dataloader_train)} loss: {current_loss:.3f} trunk_lr: {scheduler.get_last_lr()[0]:.2e} regressor_lr: {scheduler.get_last_lr()[1]:.2e}",
                    end="",
                )
                # Log training loss to TensorBoard
                writer.add_scalar(f"{client_name}/Loss/train", current_loss, global_step)
                writer.add_scalar(f"{client_name}/LR/trunk", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar(f"{client_name}/LR/regressor", scheduler.get_last_lr()[1], global_step)
                writer.add_scalar(f"{client_name}/Epoch", epoch, global_step)
                writer.add_scalar(f"{client_name}/Round", input_model.current_round, global_step)

                global_step += 1

            # Evaluate the local model after each epoch
            local_mean_test_loss, local_rmse_test_loss = evaluate(model, dataloader_test, loss_fn, device)
            # Log test loss to TensorBoard
            writer.add_scalar(f"{client_name}/Loss/local_test", local_mean_test_loss, global_step)
            writer.add_scalar(f"{client_name}/RMSE/local_test", local_rmse_test_loss, global_step)

        # End of training

        # (6) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            # Return global model performance metrics. Use negative RMSE as accuracy for global model selection.
            metrics={
                "Loss": global_mean_test_loss,
                "RMSE": global_rmse_test_loss,
                "accuracy": -1 * global_rmse_test_loss,
            },
            meta={"NUM_STEPS_CURRENT_ROUND": args.n_epochs * len(dataloader_train)},
        )
        # (7) send model back to NVFlare
        flare.send(output_model)

    # Close the TensorBoard writer
    writer.close()

    # Save the trained model
    model_save_path = os.path.join(run_dir, "fine_tuned_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Training completed and model saved to {model_save_path}")
    print(f"TensorBoard logs available in {os.path.join(run_dir, 'logs')}")

    # Save the used hyperparameters and dataset statistics
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(args.__dict__, f)


def evaluate(model, dataloader_test, loss_fn, device):
    with torch.no_grad():
        model.eval()
        test_loss = []
        for batch in dataloader_test:
            # Convert to correct dtype and move to GPU
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(torch.float32).to(device)
            labels = batch["labels"].to(torch.long).to(device)

            attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))
            output = model(input_ids, attention_mask)
            loss = loss_fn(output.squeeze(), labels)

            test_loss.append(loss.item())

        mean_test_loss = np.mean(test_loss)
        rmse_test_loss = np.sqrt(np.mean(test_loss))
        print(f"\n>>> Test MSE loss: {mean_test_loss:.3f} Test RMSE loss: {rmse_test_loss:.3f}")

    return mean_test_loss, rmse_test_loss


if __name__ == "__main__":
    main()
