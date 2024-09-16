# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import torchvision
import torchvision.transforms as transforms
from net import Net

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.app_constant import ModelName

# (optional) set a fixed location so we don't need to download everytime
CIFAR10_ROOT = "/tmp/nvflare/data"
MODEL_SAVE_PATH_ROOT = "/tmp/nvflare/data"

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=CIFAR10_ROOT, nargs="?")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?")
    parser.add_argument("--num_workers", type=int, default=1, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=2, nargs="?")
    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    local_epochs = args.local_epochs

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = Net()
    best_accuracy = 0.0

    # wraps evaluation logic into a method to re-use for
    # evaluation on both trained and received model
    def evaluate(input_weights):
        net = Net()
        net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        net.to(DEVICE)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                # (optional) use GPU to speed things up
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct // total

    # (2) initialize NVFlare client API
    flare.init()

    # (3) run continously when launch_once=true
    while flare.is_running():

        # (4) receive FLModel from NVFlare
        input_model = flare.receive()
        client_id = flare.get_site_name()

        model_path = os.path.join(MODEL_SAVE_PATH_ROOT, client_id, "cifar_net.pth")

        # Based on different "task" we will do different things
        # for "train" task (flare.is_train()) we use the received model to do training and/or evaluation
        # and send back updated model and/or evaluation metrics, if the "train_with_evaluation" is specified as True
        # in the config_fed_client we will need to do evaluation and include the evaluation metrics
        # for "evaluate" task (flare.is_evaluate()) we use the received model to do evaluation
        # and send back the evaluation metrics
        # for "submit_model" task (flare.is_submit_model()) we just need to send back the local model
        # (5) performing train task on received model
        if flare.is_train():
            print(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")

            # (5.1) loads model from NVFlare
            net.load_state_dict(input_model.params)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            # (optional) use GPU to speed things up
            net.to(DEVICE)

            if client_id == "site-1":
                local_epochs = 1
            else:
                local_epochs = 3

            steps = local_epochs * len(trainloader)

            for epoch in range(local_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    # (optional) use GPU to speed things up
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print(f"({client_id}) [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                        running_loss = 0.0

            print(f"({client_id}) Finished Training")

            # (5.2) evaluation on local trained model to save best model
            local_accuracy = evaluate(net.state_dict())
            print(f"({client_id}) Evaluating local trained model. Accuracy on the 10000 test images: {local_accuracy}")
            if local_accuracy > best_accuracy:
                best_accuracy = local_accuracy
                torch.save(net.state_dict(), model_path)

            # (5.3) evaluate on received model for model selection
            accuracy = evaluate(input_model.params)
            print(
                f"({client_id}) Evaluating received model for model selection. Accuracy on the 10000 test images: {accuracy}"
            )

            # (5.4) construct trained FL model
            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )

            # (5.5) send model back to NVFlare
            flare.send(output_model)

        # (6) performing evaluate task on received model
        elif flare.is_evaluate():
            accuracy = evaluate(input_model.params)
            print(f"({client_id}) accuracy: {accuracy}")
            flare.send(flare.FLModel(metrics={"accuracy": accuracy}))

        # (7) performing submit_model task to obtain best local model
        elif flare.is_submit_model():
            model_name = input_model.meta["submit_model_name"]
            if model_name == ModelName.BEST_MODEL:
                try:
                    weights = torch.load(model_path)
                    net = Net()
                    net.load_state_dict(weights)
                    flare.send(flare.FLModel(params=net.cpu().state_dict()))
                except Exception as e:
                    raise ValueError("Unable to load best model") from e
            else:
                raise ValueError(f"Unknown model_type: {model_name}")


if __name__ == "__main__":
    main()
