# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data.cifar10_data_utils import CIFAR10_ROOT
from data.cifar10_dataset import CIFAR10_Idx
from model import ModerateCNN

# Import nvflare client API
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper, get_lr_values

# Stream metrics to server
from nvflare.client.tracking import SummaryWriter

# Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable cuDNN auto-tuner for optimal algorithms (speeds up training with fixed input sizes)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            # (optional) use GPU to speed things up
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def create_datasets(site_name, train_idx_root, central=False):
    """To be called only after cifar10_data_split.save_split_data() downloaded the data and computed splits"""

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    if not central:
        # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
        site_idx_file_name = os.path.join(train_idx_root, site_name + ".npy")
        if os.path.exists(site_idx_file_name):
            print(f"Loading subset index for client {site_name} from {site_idx_file_name}")
            site_idx = np.load(site_idx_file_name).tolist()
        else:
            raise FileNotFoundError(f"No subset index found! File {site_idx_file_name} does not exist!")
        print(f"Client {site_name} subset size: {len(site_idx)}")
    else:
        site_idx = None  # use whole training dataset if central=True

    train_dataset = CIFAR10_Idx(
        root=CIFAR10_ROOT,
        data_idx=site_idx,
        train=True,
        download=False,
        transform=transform_train,
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=False,
        transform=transform_valid,
    )

    return train_dataset, valid_dataset


def create_data_loaders(train_dataset, valid_dataset, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, valid_loader


def compute_model_diff(model, global_model):
    """Compute the difference between local and global model weights.

    Args:
        model: The local trained model
        global_model: The global model received from server

    Returns:
        tuple: (model_diff, diff_norm) where model_diff is a dict of weight differences
               and diff_norm is the total norm of differences

    Raises:
        ValueError: If no weight differences are computed or parameters are missing
    """
    local_weights = model.state_dict()
    global_weights = global_model.state_dict()
    missing_params = []
    model_diff = {}
    diff_norm = 0.0

    for name in global_weights:
        if name not in local_weights:
            missing_params.append(name)
            continue
        # Use PyTorch operations for subtraction and convert to numpy for serialization
        model_diff[name] = local_weights[name] - global_weights[name]
        diff_norm += torch.linalg.norm(model_diff[name])

    if len(model_diff) == 0 or len(missing_params) > 0:
        raise ValueError(f"No weight differences computed or missing parameters! Missing parameters: {missing_params}")

    if torch.isnan(diff_norm) or torch.isinf(diff_norm):
        raise ValueError(f"Diff norm is NaN or Inf! Diff norm: {diff_norm}")

    print(f"Computed weight differences on {len(model_diff)} layers. Diff norm: {diff_norm}")

    return model_diff, diff_norm


def main(args):
    model = ModerateCNN()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Learning rate scheduler will be initialized after receiving first model
    # (need total_rounds for CosineAnnealingLR)
    scheduler = None

    # SCAFFOLD helper will be initialized after receiving first model if scaffold is used
    scaffold_helper = None

    if args.fedproxloss_mu > 0:
        print(f"using FedProx loss with mu {args.fedproxloss_mu}")
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    # Initializes NVFlare client API
    flare.init()
    site_name = flare.get_site_name()

    print(f"Create datasets for site {site_name}")
    train_dataset, valid_dataset = create_datasets(site_name, train_idx_root=args.train_idx_root, central=args.central)
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.central:
        # If central, we evaluate the local model on the validation set
        args.evaluate_local = True

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

        if args.scaffold:
            if scaffold_helper is None:
                scaffold_helper = PTScaffoldHelper()
                scaffold_helper.init(model=model)  # Initialize the SCAFFOLD helper after moving model to GPU device
            if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL not in input_model.meta:
                raise ValueError(
                    f"Expected model meta to contain AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL "
                    f"but meta was {input_model.meta}.",
                )
            global_ctrl_weights = input_model.meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
            if not global_ctrl_weights:
                raise ValueError("global_ctrl_weights were empty!")
            # convert to tensor and load into c_global model
            for k in global_ctrl_weights.keys():
                global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k]).to(DEVICE)
            scaffold_helper.load_global_controls(weights=global_ctrl_weights)

            # local_train with SCAFFOLD steps
            c_global_para, c_local_para = scaffold_helper.get_params()

        # Evaluate on received global model for model selection
        val_acc_global_model = evaluate(global_model, valid_loader)
        print(f"Global model accuracy on validation set: {100 * val_acc_global_model:.2f} %")
        summary_writer.add_scalar(
            tag="val_acc_global_model", scalar=val_acc_global_model, global_step=input_model.current_round
        )

        # Calculate total steps
        steps = args.aggregation_epochs * len(train_loader)
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

                curr_lr = get_lr_values(optimizer)[0]
                if args.scaffold:
                    # SCAFFOLD step
                    scaffold_helper.model_update(
                        model=model, curr_lr=curr_lr, c_global_para=c_global_para, c_local_para=c_local_para
                    )
                # accumulate loss
                running_loss += loss.item()

            # Report loss at the end of each epoch
            avg_loss = running_loss / len(train_loader)
            global_epoch = input_model.current_round * args.aggregation_epochs + epoch
            summary_writer.add_scalar(tag="global_round", scalar=input_model.current_round, global_step=global_epoch)
            summary_writer.add_scalar(tag="global_epoch", scalar=global_epoch, global_step=global_epoch)
            print(
                f"{site_name}: Epoch [{epoch + 1}/{args.aggregation_epochs}] - Average Loss: {avg_loss:.4f} - Learning Rate: {curr_lr:.6f}"
            )
            summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=global_epoch)
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

        if args.scaffold:
            # Update the SCAFFOLD terms
            scaffold_helper.terms_update(
                model=model,
                curr_lr=curr_lr,
                c_global_para=c_global_para,
                c_local_para=c_local_para,
                model_global=global_model,
            )

        print(f"Finished training for current round {input_model.current_round}")

        # compute delta model, global model has the primary key set
        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar(tag="diff_norm", scalar=diff_norm.item(), global_step=input_model.current_round)

        # Construct trained FL model
        meta = {"NUM_STEPS_CURRENT_ROUND": steps}
        if args.scaffold:
            # Add scaffold controls to resulting model
            meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF] = scaffold_helper.get_delta_controls()

        output_model = flare.FLModel(
            params=model_diff.cpu(),
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
        "--no_lr_scheduler", action="store_true", help="Whether to use a learning rate scheduler. Default is False."
    )
    parser.add_argument(
        "--cosine_lr_eta_min_factor",
        type=float,
        default=0.01,
        help="Minimum learning rate as a factor of the initial learning rate for cosine annealing scheduler. Default is 0.01 (or 1%).",
    )
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0, help="FedProx loss mu. Default is 0.")
    parser.add_argument(
        "--central", action="store_true", help="Whether to centralized dataset for training. Default is False."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading. Default is 2.")
    parser.add_argument("--scaffold", action="store_true", help="Whether to use scaffold. Default is False.")
    parser.add_argument(
        "--evaluate_local",
        action="store_true",
        help="Whether to evaluate the local model on the validation set. Default is False.",
    )
    args = parser.parse_args()

    main(args)
