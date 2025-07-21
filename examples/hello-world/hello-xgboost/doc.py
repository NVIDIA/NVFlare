"""

**Hello Pytorch** ||
`hello lightning <../hello-lightning/tutorial.html>`_ ||
`hello tensorflow <../hello-tf/tutorial.html>`_ ||
`hello LR <../hello-lr/tutorial.html>`_ ||
`hello KMeans <../hello-KMeans/tutorial.html>`_ ||
`hello KM <../hello-km/tutorial.html>`_ ||
`hello stats <../hello-stats/tutorial.html>`_ ||
`hello cyclic <../hello-cyclic/tutorial.html>`_ ||
`hello-xgboost <../hello-xgboost/tutorial.html>`_ ||
`hello-flower <../hello-flower/tutorial.html>`_ ||


Hello Pytorch
===================

This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using
federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>`.
It is recommended to create a virtual environment and run everything within a virtualenv.

NVIDIA FLARE Installation
-------------------------
for the complete installation instructions, see <../../installation.html>
pip install nvflare

Code Structure
--------------
first get the example code from github:
git clone https://github.com/NVIDIA/NVFlare.git
then navigate to the hello-pt directory:

git switch <release branch>
cd examples/hello-world/hello-pt

hello-pt

.. code-block:: text

    .
    |
    |-- src
    |    |
    |    |-- client.py        # client local training script
    |    |-- model.py         # model definition
    |-- job.py              # job recipe that defines client and server configurations
    |-- requirements.txt    # dependencies
    |-- doc.py              # documentation file to generate the current documentation

doc.py is this file, not part of the fl code.

Data
-----------------
This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module,
You can split the datasets for different clients, so that each client has its own dataset.
Here for simplicity's sake, the same dataset we will be using on each client.

"""
################################
# Model
# ------------------
# In PyTorch, neural networks are implemented by defining a class (e.g., `SimpleNetwork`) that extends
# `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. The network’s architecture is
# set up in the `__init__` method,# while the `forward` method determines how input data flows through the layers.
# For faster computations, the model is transferred to a hardware accelerator (such as CUDA GPUs) if available;
# otherwise, it runs on the CPU. The implementation of this model can be found in model.py.

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = SimpleNetwork().to(device)
print(model)

######################################################################
# --------------
#


#####################################################################
# Client Code
# ------------------
#
# The client code is responsible for
# Notice the training code is almost identical to the pytorch standard training code.
# The only difference is that we added a few lines to receive and send data to the server.
#

import os

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# (1) import nvflare client API
import nvflare.client as flare
# (2) import nvflare experimental tracking API for metrics collection, this is optional
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/tmp/nvflare/data"

def main():
    batch_size = 4
    epochs = 5
    lr = 0.01
    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # (3) import init FLARE API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    train_dataset = CIFAR10(
        root=os.path.join(DATASET_PATH, client_name), transform=transforms, download=True, train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    summary_writer = SummaryWriter()
    # (4) if the overall FL training loop is still running
    while flare.is_running():
        # (5) receive global model from FL Server. This is the aggregated model
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    print(f"Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss / 3000}")
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="loss_for_each_batch", scalar=running_loss, global_step=global_step)
                    running_loss = 0.0

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        # (6) prepare new model update
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        # (7) send model back to NVFlare
        flare.send(output_model)

#####################################################################
# Server Code
# ------------------
# In federated averaging, the server code is responsible for
# aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
# In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
# The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
# There is no need to defined a customized server code for this example.


#####################################################################
# Job Recipe Code
# ------------------
#


#####################################################################
# Run FL Job
# ------------------
#

# python job.py
