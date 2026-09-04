# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Client-side training for the Hello PyTorch example."""

import argparse

import torch
from model import create_model
from prepare_data import DATASET_CHOICES, DATASET_PATH, DEFAULT_DATASET, SyntheticImageDataset, stable_seed
from torch import nn
from torch.optim import SGD

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

LOCAL_MODEL_PATH = "./local_model.pt"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CIFAR_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 1
DEFAULT_NUM_WORKERS = 0
DEFAULT_TEST_SIZE = 100
DEFAULT_TRAIN_SIZE = 200
DEFAULT_SYNTHETIC_LEARNING_RATE = 0.1


def evaluate(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            # (optional) use GPU to speed things up
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if total == 0:
            raise ValueError("Evaluation data_loader produced no samples; check data preparation and --test_size.")
        accuracy = 100.0 * correct / total
        print(f"Accuracy of the network on {total} test images: {accuracy:.2f}%")
    return accuracy


def create_data_loaders(dataset, site_name, train_size, test_size, batch_size, num_workers, data_root=DATASET_PATH):
    if dataset == "synthetic":
        train_set = SyntheticImageDataset(site_name=site_name, split="train", size=train_size)
        test_set = SyntheticImageDataset(site_name=site_name, split="eval", size=test_size)
    else:
        import torchvision
        from torchvision.transforms import Compose, Normalize, ToTensor

        # Simulation clients share this cache. ``prepare_data.py`` downloads
        # CIFAR-10 once before clients start, avoiding concurrent writes.
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    shuffle_generator = torch.Generator().manual_seed(stable_seed(site_name, "train-loader"))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=shuffle_generator,
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", choices=DATASET_CHOICES, dest="dataset")
    dataset_group.add_argument(
        "--synthetic_data",
        action="store_const",
        const="synthetic",
        dest="dataset",
        help="Deprecated alias for --dataset synthetic.",
    )
    parser.set_defaults(dataset=DEFAULT_DATASET)
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--data_root",
        default=DATASET_PATH,
        help="Client-local CIFAR-10 cache path. Ignored for the synthetic dataset.",
    )
    args = parser.parse_args()
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = DEFAULT_SYNTHETIC_LEARNING_RATE if args.dataset == "synthetic" else DEFAULT_CIFAR_LEARNING_RATE

    # (3) initializes NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # Keep the zero-argument quickstart on CPU for repeatable behavior across
    # developer machines. The opt-in CIFAR-10 run can use a GPU when available.
    device = torch.device("cpu" if args.dataset == "synthetic" else "cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loader, test_loader = create_data_loaders(
        dataset=args.dataset,
        site_name=client_name,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )
    last_params = None

    # The client writer is transport-only and does not require TensorBoard.
    # An optional server-side tracking receiver decides whether to persist it.
    summary_writer = SummaryWriter()

    while flare.is_running():
        # (4) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        # Cross-site evaluation requests the client's latest local model without
        # sending model parameters in the request.
        if flare.is_submit_model():
            if last_params is None:
                error_msg = "submit_model called before a local model was trained"
                print(f"ERROR: {error_msg}")
                # TaskScriptRunner converts this exception into TOPIC_ABORT so the
                # executor can report the task failure instead of waiting for a result.
                raise RuntimeError(error_msg)
            print(f"site = {client_name}, submitting local model")
            flare.send(flare.FLModel(params=last_params))
            continue

        # (5) loads model from NVFlare
        model.load_state_dict(input_model.params)
        # (6) evaluate the received global model before local training
        accuracy_before_training = evaluate(model, test_loader, device)

        # (optional) Task branch for cross-site evaluation
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")
            # For CSE, just return the evaluation metrics without training
            output_model = flare.FLModel(metrics={"accuracy": accuracy_before_training})
            flare.send(output_model)
            continue

        model.train()
        steps = args.epochs * len(train_loader)
        for epoch in range(args.epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
            avg_loss = running_loss / len(train_loader)
            print(f"site={client_name}, epoch={epoch + 1}/{args.epochs}, loss={avg_loss:.4f}")
            global_step = input_model.current_round * args.epochs + epoch
            summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=global_step)

        print(f"Finished Training for {client_name}")
        trained_accuracy = evaluate(model, test_loader, device)

        last_params = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
        torch.save(last_params, LOCAL_MODEL_PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=last_params,
            # The primary metric evaluates the received global model, which is
            # the model the server considers for best-model selection. Report
            # the trained local model separately to make progress visible.
            metrics={
                "accuracy": accuracy_before_training,
                "accuracy_after_local_training": trained_accuracy,
            },
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
