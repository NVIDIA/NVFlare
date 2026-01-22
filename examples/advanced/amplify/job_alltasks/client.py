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
from utils import load_and_validate_csv

# (1) import nvflare client API
import nvflare.client as flare


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AMPLIFY model for sequence regression")
    # Data paths
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for training and test data")
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help="List of task names for each regressor")
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
    # Data sampling (for quick testing)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from train/test sets (default: None, use all samples). Useful for quick testing.",
    )
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

    # (2) initializes NVFlare client API
    flare.init()
    client_name = flare.get_site_name()

    print(f"Client {client_name} training on all {len(args.tasks)} tasks: {args.tasks}")

    # Set random seed for deterministic training
    set_seed(args.seed)

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

    # Build Regression model on top of AMPLIFY with multiple regressors
    model = AmplifyRegressor(
        pretrained_model_name_or_path=args.pretrained_model,
        layer_sizes=layer_sizes,
        num_targets=len(args.tasks),  # One regressor per task/CSV file
    )

    # Print model architecture and configuration
    print_model_info(model, layer_sizes, args)

    # Load AMPLIFY tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)

    # Create dataloaders for each CSV file pair
    dataloaders_train = []
    dataloaders_test = []

    for task in args.tasks:
        # Convert NVFlare site name (e.g., "site-1") to client file format (e.g., "client1")
        # NVFlare site names: "site-1", "site-2", etc.
        # Data file names: "client1_train_data.csv", "client2_train_data.csv", etc.
        if client_name.startswith("site-"):
            client_id = client_name.split("-")[1]  # Extract number from "site-N"
            file_client_name = f"client{client_id}"
        else:
            file_client_name = client_name
        
        # Construct data paths based on client name and task
        train_csv = os.path.join(args.data_root, task, f"{file_client_name}_train_data.csv")
        test_csv = os.path.join(args.data_root, task, "test_data.csv")
        
        print(f"  Task {task}:")
        
        # Load and validate dataset
        dataset = load_and_validate_csv(train_csv, test_csv, verbose=False, max_samples=args.max_samples, seed=args.seed)
        print(f"    Loaded: Train samples={len(dataset['train'])}, Test samples={len(dataset['test'])}")

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

        dataloaders_train.append(dataloader_train)
        dataloaders_test.append(dataloader_test)

    # Build the loss, optimizer, and scheduler
    loss_fn = torch.nn.MSELoss()

    # Create parameter groups with different learning rates
    param_groups = [
        {"params": model.trunk.parameters(), "lr": args.trunk_lr},
        {"params": model.regressors.parameters(), "lr": args.regressor_lr},
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
        total_iters=len(dataloaders_train[0]) * (args.n_epochs) * (total_rounds - 1),
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

        # (5) Evaluate the global model on all test sets
        global_metrics = []
        for regressor_idx, (dataloader_test, task_name) in enumerate(zip(dataloaders_test, args.tasks)):
            global_mean_test_loss, global_rmse_test_loss, global_pearson_corr = evaluate(
                model, dataloader_test, loss_fn, device, regressor_idx=regressor_idx, task_name=task_name
            )
            global_metrics.append(
                {"mean_loss": global_mean_test_loss, "rmse": global_rmse_test_loss, "pearson": global_pearson_corr}
            )
            # Log test metrics to TensorBoard
            writer.add_scalar(f"{client_name}/Loss/global_test_{task_name}", global_mean_test_loss, global_step)
            writer.add_scalar(f"{client_name}/RMSE/global_test_{task_name}", global_rmse_test_loss, global_step)
            writer.add_scalar(f"{client_name}/Pearson/global_test_{task_name}", global_pearson_corr, global_step)

        # Training loop
        for epoch in range(args.n_epochs):
            train_losses = [[] for _ in range(len(dataloaders_train))]
            model.train()

            # Train each regressor with its corresponding dataloader
            for regressor_idx, (dataloader_train, task_name) in enumerate(zip(dataloaders_train, args.tasks)):
                for i, batch in enumerate(dataloader_train):
                    # Convert to correct dtype and move to GPU
                    input_ids = batch["input_ids"].to(torch.long).to(device)
                    attention_mask = batch["attention_mask"].to(torch.float32).to(device)
                    labels = batch["labels"].to(device)

                    # Convert the attention mask to an additive mask
                    attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))

                    # Compute the model output for the specific regressor
                    output = model(
                        input_ids, attention_mask, frozen_trunk=args.frozen_trunk, regressor_idx=regressor_idx
                    )

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
                    train_losses[regressor_idx].append(loss.item())
                    current_loss = np.mean(train_losses[regressor_idx])
                    print(
                        f"\rTask {task_name} - Epoch: {epoch:6d}/{args.n_epochs - 1} Step {i:6d}/{len(dataloader_train)} loss: {current_loss:.3f} trunk_lr: {scheduler.get_last_lr()[0]:.2e} regressor_lr: {scheduler.get_last_lr()[1]:.2e}",
                        end="",
                    )
                    # Log training metrics to TensorBoard
                    writer.add_scalar(f"{client_name}/Loss/train_{task_name}", current_loss, global_step)
                    writer.add_scalar(f"{client_name}/LR/trunk", scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar(f"{client_name}/LR/regressor", scheduler.get_last_lr()[1], global_step)
                    writer.add_scalar(f"{client_name}/Epoch", epoch, global_step)
                    writer.add_scalar(f"{client_name}/Round", input_model.current_round, global_step)

                    global_step += 1

            # Evaluate the local model after each epoch
            local_metrics = []
            for regressor_idx, (dataloader_test, task_name) in enumerate(zip(dataloaders_test, args.tasks)):
                local_mean_test_loss, local_rmse_test_loss, local_pearson_corr = evaluate(
                    model, dataloader_test, loss_fn, device, regressor_idx=regressor_idx, task_name=task_name
                )
                local_metrics.append(
                    {"mean_loss": local_mean_test_loss, "rmse": local_rmse_test_loss, "pearson": local_pearson_corr}
                )
                # Log test metrics to TensorBoard
                writer.add_scalar(f"{client_name}/Loss/local_test_{task_name}", local_mean_test_loss, global_step)
                writer.add_scalar(f"{client_name}/RMSE/local_test_{task_name}", local_rmse_test_loss, global_step)
                writer.add_scalar(f"{client_name}/Pearson/local_test_{task_name}", local_pearson_corr, global_step)

        # End of training

        # Calculate average metrics across all regressors
        avg_global_rmse = np.mean([m["rmse"] for m in global_metrics])
        avg_global_pearson = np.mean([m["pearson"] for m in global_metrics])

        # (6) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            # Return global model performance metrics. Use negative RMSE as accuracy for global model selection.
            metrics={
                "Loss": np.mean([m["mean_loss"] for m in global_metrics]),
                "RMSE": avg_global_rmse,
                "Pearson": avg_global_pearson,
                "accuracy": -1 * avg_global_rmse,
            },
            meta={"NUM_STEPS_CURRENT_ROUND": args.n_epochs * len(dataloaders_train[0])},
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


def evaluate(model, dataloader_test, loss_fn, device, regressor_idx=None, task_name=None):
    with torch.no_grad():
        model.eval()
        test_loss = []
        all_predictions = []
        all_labels = []
        for batch in dataloader_test:
            # Convert to correct dtype and move to GPU
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(torch.float32).to(device)
            labels = batch["labels"].to(torch.long).to(device)

            attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf"))
            output = model(input_ids, attention_mask, regressor_idx=regressor_idx)
            loss = loss_fn(output.squeeze(), labels)

            test_loss.append(loss.item())

            # Store predictions and labels for correlation calculation
            all_predictions.extend(output.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        mean_test_loss = np.mean(test_loss)
        rmse_test_loss = np.sqrt(np.mean(test_loss))

        # Calculate Pearson correlation
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        pearson_corr = np.corrcoef(all_predictions, all_labels)[0, 1]

        print(
            f"\n>>> Task {task_name} - Test MSE loss: {mean_test_loss:.3f} Test RMSE loss: {rmse_test_loss:.3f} Pearson correlation: {pearson_corr:.3f}"
        )

    return mean_test_loss, rmse_test_loss, pearson_corr


if __name__ == "__main__":
    main()