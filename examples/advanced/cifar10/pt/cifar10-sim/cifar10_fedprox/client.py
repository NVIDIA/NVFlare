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
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10_data_utils import create_data_loaders, create_datasets
from model import ModerateCNN
from train_utils import compute_model_diff, evaluate, get_lr_values

# Import nvflare client API
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss

# Stream metrics to server
from nvflare.client.tracking import SummaryWriter

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

    # Learning rate scheduler will be initialized after receiving first model
    # (need total_rounds for CosineAnnealingLR)
    scheduler = None

    if args.fedproxloss_mu > 0:
        print(f"using FedProx loss with mu {args.fedproxloss_mu}")
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    # Initializes NVFlare client API
    flare.init()
    site_name = flare.get_site_name()

    print(f"Create datasets for site {site_name}")
    train_dataset, valid_dataset = create_datasets(site_name, train_idx_root=args.train_idx_root)
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    summary_writer = SummaryWriter()
    while flare.is_running():
        # Receive FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

        # Initialize cosine annealing scheduler on first round
        if scheduler is None and not args.no_lr_scheduler:
            total_rounds = input_model.total_rounds
            eta_min = args.lr * args.cosine_lr_eta_min_factor  # Minimum LR is a factor of the initial LR
            T_max = total_rounds * args.aggregation_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            print(
                f"{site_name}: Using CosineAnnealingLR scheduler: initial_lr={args.lr}, eta_min={eta_min}, T_max={T_max}"
            )

        # Load model from NVFlare
        model.load_state_dict(input_model.params, strict=True)  # make sure all parameters are loaded with strict mode

        # Copy global model for computing weight differences and for FedProx loss
        global_model = copy.deepcopy(model)
        for param in global_model.parameters():
            param.requires_grad = False

        # Use GPU to speed things up
        model.to(DEVICE)
        global_model.to(DEVICE)

        # Evaluate on received global model for model selection
        val_acc_global_model = evaluate(global_model, valid_loader)
        print(f"Global model accuracy on validation set: {100 * val_acc_global_model:.2f} %")
        summary_writer.add_scalar(
            tag="val_acc_global_model", scalar=val_acc_global_model, global_step=input_model.current_round
        )

        # Calculate total steps
        steps = args.aggregation_epochs * len(train_loader)
        curr_lr = get_lr_values(optimizer)[0]  # Initialize learning rate before training loop
        for epoch in range(args.aggregation_epochs):  # loop over the dataset multiple times
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # (optional) use GPU to speed things up
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if args.fedproxloss_mu > 0:
                    fed_prox_loss = criterion_prox(model, global_model)
                    loss += fed_prox_loss

                loss.backward()
                optimizer.step()

                # accumulate loss
                running_loss += loss.item()

            # Report loss at the end of each epoch
            avg_loss = running_loss / len(train_loader)
            global_epoch = input_model.current_round * args.aggregation_epochs + epoch
            summary_writer.add_scalar(tag="global_round", scalar=input_model.current_round, global_step=global_epoch)
            summary_writer.add_scalar(tag="global_epoch", scalar=global_epoch, global_step=global_epoch)
            summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=global_epoch)
            curr_lr = get_lr_values(optimizer)[0]  # Get current learning rate
            print(
                f"{site_name}: Epoch [{epoch + 1}/{args.aggregation_epochs}] - Average Loss: {avg_loss:.4f} - Learning Rate: {curr_lr:.6f}"
            )
            summary_writer.add_scalar(tag="learning_rate", scalar=curr_lr, global_step=global_epoch)

            # Optionally evaluate the current local model on validation set only
            if args.evaluate_local:
                val_acc_local_model = evaluate(model, valid_loader)
                print(f"Local model accuracy on validation set: {100 * val_acc_local_model:.2f} %")
                summary_writer.add_scalar(
                    tag="val_acc_local_model", scalar=val_acc_local_model, global_step=global_epoch
                )

            # Step the learning rate scheduler at the end of each epoch
            if scheduler is not None:
                scheduler.step()

        print(f"Finished training for current round {input_model.current_round}")

        # compute delta model, global model has the primary key set
        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar(tag="diff_norm", scalar=diff_norm.item(), global_step=input_model.current_round)

        # Construct trained FL model
        meta = {"NUM_STEPS_CURRENT_ROUND": steps}

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": val_acc_global_model},
            meta=meta,
        )

        # Send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_idx_root",
        type=str,
        default="/tmp/cifar10_splits",
        help="Root directory for training index. Default is /tmp/cifar10_splits.",
    )
    parser.add_argument(
        "--aggregation_epochs",
        type=int,
        default=4,
        help="Number of local training epochs per client and FL round. Default is 4.",
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
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0, help="FedProx loss mu. Default is 0.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading. Default is 2.")
    parser.add_argument(
        "--evaluate_local",
        action="store_true",
        help="Whether to evaluate the local model on the validation set. Default is False.",
    )
    args = parser.parse_args()

    main(args)
