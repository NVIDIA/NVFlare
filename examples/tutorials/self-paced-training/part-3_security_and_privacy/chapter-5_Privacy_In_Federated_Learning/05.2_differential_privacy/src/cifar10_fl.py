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

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net
from opacus import PrivacyEngine

# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# (optional) set a fix place so we don't need to download everytime
DATASET_PATH = "/tmp/nvflare/data"
# If available, we use GPU to speed things up.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device {DEVICE}")


def main(target_epsilon, max_grad_norm):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    epochs = 1

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    # (2) initializes NVFlare client API
    flare.init()

    # (Optional) compute unique seed from client name to initialize data loaders
    client_name = flare.get_site_name()
    seed = int.from_bytes(client_name.encode(), "big")
    torch.manual_seed(seed)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Optionally add DP engine
    if target_epsilon:
        target_delta = 1e-5  # 1/(len(trainloader)*batch_size) # "The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset" (from https://opacus.ai/tutorials/building_image_classifier).
        print(f"Adding privacy engine with epsilon={target_epsilon}, delta={target_delta}")
        privacy_engine = PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs * flare.receive().total_rounds,
            max_grad_norm=max_grad_norm,
        )

    summary_writer = SummaryWriter()
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")

        # (4) loads model from NVFlare
        if target_epsilon:
            # Opacus adds "_module." prefix to state dict. Add it here to match local state dict
            global_params = {}
            for k, v in input_model.params.items():
                global_params[f"_module.{k}"] = v
        else:
            global_params = input_model.params
        net.load_state_dict(global_params)

        # (optional) use GPU to speed things up
        net.to(DEVICE)
        net.train()
        # (optional) calculate total steps
        steps = epochs * len(trainloader)
        for epoch in range(epochs):  # loop over the dataset multiple times

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
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                    global_step = input_model.current_round * steps + epoch * len(trainloader) + i

                    summary_writer.add_scalar(
                        tag="loss_for_each_batch", scalar=running_loss / 100, global_step=global_step
                    )
                    running_loss = 0.0

                    if target_epsilon:
                        epsilon = privacy_engine.get_epsilon(target_delta)
                        print(f"Training with privacy (ε = {epsilon:.2f}, δ = {target_delta})")

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(net.state_dict(), PATH)

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            net = Net()
            net.load_state_dict(input_weights)
            # (optional) use GPU to speed things up
            net.to(DEVICE)
            net.eval()

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

            print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
            return 100 * correct // total

        # (6) evaluate on received model for model selection
        accuracy = evaluate(input_model.params)
        summary_writer.add_scalar(tag="global_model_accuracy", scalar=accuracy, global_step=input_model.current_round)
        # (7) construct trained FL model

        if target_epsilon:
            # Remove prefix added by Opacus again to match global state dict
            local_params = {}
            for k, v in net.cpu().state_dict().items():
                local_params[k.replace("_module.", "")] = v
        else:
            local_params = net.cpu().state_dict()

        output_model = flare.FLModel(
            params=local_params,
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_epsilon", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    main(target_epsilon=args.target_epsilon, max_grad_norm=args.max_grad_norm)
