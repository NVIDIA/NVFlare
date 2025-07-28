
"""
`hello pytorch <../hello-pt/doc.html>`_ ||
`hello lightning <../hello-lightning/doc.html>`_ ||
`hello tensorflow <../hello-tf/doc.html>`_ ||
`hello LR <../hello-lr/doc.html>`_ ||
`hello KMeans <../hello-kmeans/doc.html>`_ ||
`hello KM <../hello-km/doc.html>`_ ||
`hello stats <../hello-stats/doc.html>`_ ||
**hello cyclic** ||
`hello-xgboost <../hello-xgboost/doc.html>`_ ||
`hello-flower <../hello-flower/doc.html>`_ ||


Hello Cyclic
===================

This example demonstrates how to use NVIDIA FLARE with **Tensorflow** to train an image classifier using
cyclic weight transfer approach.The complete example code can be found in the`hello-cyclic directory <examples/hello-world/hello-cyclic/>`_.
It is recommended to create a virtual environment and run everything within a virtualenv.

NVIDIA FLARE Installation
-------------------------
for the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare

Install the dependency

.. code-block:: text

    pip install -r requirements.txt


Code Structure
--------------

first get the example code from github:

.. code-block:: text

    git clone https://github.com/NVIDIA/NVFlare.git

then navigate to the hello-pt directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-cyclic


.. code-block:: text

    hello-pt
        |
        |-- client.py         # client local training script
        |-- model.py          # model definition
        |-- job.py            # job recipe that defines client and server configurations
        |-- requirements.txt  # dependencies

Data
-----------------
This example uses the `MNIST dataset`

.. code-block:: text

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = (
        train_images / 255.0,
        test_images / 255.0,
    )

    # simulate separate datasets for each client by dividing MNIST dataset in half
    client_name = flare.get_site_name()
    if client_name == "site-1":
        train_images = train_images[: len(train_images) // 2]
        train_labels = train_labels[: len(train_labels) // 2]
        test_images = test_images[: len(test_images) // 2]
        test_labels = test_labels[: len(test_labels) // 2]
    elif client_name == "site-2":
        train_images = train_images[len(train_images) // 2 :]
        train_labels = train_labels[len(train_labels) // 2 :]
        test_images = test_images[len(test_images) // 2 :]
        test_labels = test_labels[len(test_labels) // 2 :]




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

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/tmp/nvflare/data"


def main():
    batch_size = 16
    epochs = 2
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

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    train_dataset = CIFAR10(
        root=os.path.join(DATASET_PATH, client_name), transform=transforms, download=True, train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

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
                    print(f"site={client_name}, Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss / 3000}")
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="loss_for_each_batch", scalar=running_loss, global_step=global_step)
                    running_loss = 0.0

        print(f"Finished Training for {client_name}")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    main()

#####################################################################
# Server Code
# ------------------
# In federated averaging, the server code is responsible for
# aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
# In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
# The Cyclic class is defined in `nvflare.app_common.workflows.cyclic import Cyclic`
# There is no need to define a customized server code for this example.


#####################################################################
# Job Recipe Code
# ------------------
# Job Recipe contains the client.py and built-in cyclic algorithm.
from model import Net

from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.job_config.cyclic_recipe import CyclicRecipe
from nvflare.job_config.script_runner import FrameworkType

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "client.py"

    recipe = CyclicRecipe(
        framework=FrameworkType.TENSORFLOW,
        min_clients=1,
        num_rounds=num_rounds,
        model=TFModel(Net()),
        client_script=train_script,
        client_script_args="",
    )

    recipe.execute(clients=n_clients)


#####################################################################
# Run FL Job
# ------------------
#
# This section provides the command to execute the federated learning job
# using the job recipe defined above. Run this command in your terminal.


#####################################################################
# **Command to execute the FL job**
#
# Use the following command in your terminal to start the job with the specified
# number of rounds, batch size, and number of clients.
#
#
# .. code-block:: text
#
#   python job.py


