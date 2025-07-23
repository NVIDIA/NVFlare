
"""
**Hello Pytorch** ||
`hello lightning <../hello-lightning/doc.html>`_ ||
`hello tensorflow <../hello-tf/doc.html>`_ ||
`hello LR <../hello-lr/doc.html>`_ ||
`hello KMeans <../hello-kmeans/doc.html>`_ ||
`hello KM <../hello-km/doc.html>`_ ||
`hello stats <../hello-stats/doc.html>`_ ||
`hello cyclic <../hello-cyclic/doc.html>`_ ||
`hello-xgboost <../hello-xgboost/doc.html>`_ ||
`hello-flower <../hello-flower/doc.html>`_ ||


Hello Pytorch
===================

This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using
federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>`.
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
    cd examples/hello-world/hello-pt


.. code-block:: text

    hello-pt
        |
        |-- client.py         # client local training script
        |-- model.py          # model definition
        |-- job.py            # job recipe that defines client and server configurations
        |-- requirements.txt  # dependencies
        |-- doc.py            # documentation file to generate the current documentation

doc.py is this file, it is used for documentation generation, it is not part of the fl code.

Data
-----------------
This example uses the `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset

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
# The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
# There is no need to defined a customized server code for this example.


#####################################################################
# Job Recipe Code
# ------------------
# Job Recipe contains the client.py and built-in fedavg algorithm.

from nvflare.app_opt.pt.job_config.Job_recipe import FedAvgRecipe

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/client.py"
    client_script_args = ""

    recipe = FedAvgRecipe(clients=n_clients,
                          num_rounds=num_rounds,
                          model= SimpleNetwork(),
                          client_script=train_script,
                          client_script_args= client_script_args)
    recipe.execute()



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
#   python job.py --num_rounds 2 --batch_size 16


#####################################################################
# output
#
# .. code-block:: text
#
#     2025-07-20 21:41:23,489 - INFO - model selection weights control: {}
#     2025-07-20 21:41:24,357 - INFO - Tensorboard records can be found in /tmp/nvflare/sim/workspace/server/simulate_job/tb_events you can view it using `tensorboard --logdir=/tmp/nvflare/sim/workspace/server/simulate_job/tb_events`
#     2025-07-20 21:41:24,358 - INFO - Initializing BaseModelController workflow.
#     2025-07-20 21:41:24,359 - INFO - Beginning model controller run.
#     2025-07-20 21:41:24,359 - INFO - Start FedAvg.
#     2025-07-20 21:41:24,360 - INFO - loading initial model from persistor
#     2025-07-20 21:41:24,360 - INFO - Both source_ckpt_file_full_name and ckpt_preload_path are not provided. Using the default model weights initialized on the persistor side.
#     2025-07-20 21:41:24,361 - INFO - Round 0 started.
#     2025-07-20 21:41:24,361 - INFO - Sampled clients: ['site-1', 'site-2']
#     2025-07-20 21:41:24,361 - INFO - Sending task train to ['site-1', 'site-2']
#     2025-07-20 21:41:28,049 - INFO - start task run() with full path: /tmp/nvflare/sim/workspace/site-2/simulate_job/app_site-2/custom/client.py
#     2025-07-20 21:41:28,052 - INFO - start task run() with full path: /tmp/nvflare/sim/workspace/site-1/simulate_job/app_site-1/custom/client.py
#     2025-07-20 21:41:28,060 - INFO - execute for task (train)
#     2025-07-20 21:41:28,060 - INFO - send data to peer
#     2025-07-20 21:41:28,061 - INFO - sending payload to peer
#     2025-07-20 21:41:28,061 - INFO - Waiting for result from peer
#     2025-07-20 21:41:28,063 - INFO - execute for task (train)
#     2025-07-20 21:41:28,063 - INFO - send data to peer
#     2025-07-20 21:41:28,064 - INFO - sending payload to peer
#     2025-07-20 21:41:28,064 - INFO - Waiting for result from peer
#     2025-07-20 21:41:29,128 - INFO - Files already downloaded and verified
#     2025-07-20 21:41:29,168 - INFO - Files already downloaded and verified
#     2025-07-20 21:41:29,566 - INFO - site = site-1, current_round=0
#     2025-07-20 21:41:29,599 - INFO - site = site-2, current_round=0
#     2025-07-20 21:41:29,987 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 4.805266360441844e-05
#     2025-07-20 21:41:30,002 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 4.810774823029836e-05
#     2025-07-20 21:41:39,339 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.10355195295189817
#     2025-07-20 21:41:39,348 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.1036934811597069
#     2025-07-20 21:41:39,737 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.9984918733437858e-05
#     2025-07-20 21:41:39,746 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 2.9017041126887005e-05
#     2025-07-20 21:41:49,111 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.08605644813800852
#     2025-07-20 21:41:49,120 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.08581393839791417
#     2025-07-20 21:41:49,497 - INFO - Finished Training for site-1
#     2025-07-20 21:41:49,506 - INFO - Finished Training for site-2
#     2025-07-20 21:41:49,507 - INFO - site: site-1, sending model to server.
#     2025-07-20 21:41:49,508 - INFO - site: site-2, sending model to server.
#     2025-07-20 21:41:49,990 - INFO - aggregating 2 update(s) at round 0
#     2025-07-20 21:41:49,992 - INFO - Start persist model on server.
#     2025-07-20 21:41:49,995 - INFO - End persist model on server.
#     2025-07-20 21:41:49,995 - INFO - Round 1 started.
#     2025-07-20 21:41:49,995 - INFO - Sampled clients: ['site-1', 'site-2']
#     2025-07-20 21:41:49,996 - INFO - Sending task train to ['site-1', 'site-2']
#     2025-07-20 21:41:51,695 - INFO - execute for task (train)
#     2025-07-20 21:41:51,696 - INFO - send data to peer
#     2025-07-20 21:41:51,696 - INFO - sending payload to peer
#     2025-07-20 21:41:51,697 - INFO - Waiting for result from peer
#     2025-07-20 21:41:51,754 - INFO - execute for task (train)
#     2025-07-20 21:41:51,754 - INFO - send data to peer
#     2025-07-20 21:41:51,755 - INFO - sending payload to peer
#     2025-07-20 21:41:51,756 - INFO - Waiting for result from peer
#     2025-07-20 21:41:52,010 - INFO - site = site-1, current_round=1
#     2025-07-20 21:41:52,011 - INFO - site = site-2, current_round=1
#     2025-07-20 21:41:52,023 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 3.0566091338793434e-05
#     2025-07-20 21:41:52,023 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 2.9468193650245666e-05
#     2025-07-20 21:42:05,812 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.08019066233622531
#     2025-07-20 21:42:06,095 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.08063799119119842
#     2025-07-20 21:42:06,218 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 2.4138346314430236e-05
#     2025-07-20 21:42:06,494 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.6129357516765596e-05
#     2025-07-20 21:42:15,555 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.07604475673598547
#     2025-07-20 21:42:15,841 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.07583552108642956
#     2025-07-20 21:42:15,944 - INFO - Finished Training for site-2
#     2025-07-20 21:42:15,945 - INFO - site: site-2, sending model to server.
#     2025-07-20 21:42:16,217 - INFO - Finished Training for site-1
#     2025-07-20 21:42:16,220 - INFO - site: site-1, sending model to server.
#     2025-07-20 21:42:16,236 - WARNING - validation metric not existing in site-1
#     2025-07-20 21:42:16,300 - WARNING - validation metric not existing in site-2
#     2025-07-20 21:42:16,427 - INFO - aggregating 2 update(s) at round 1
#     2025-07-20 21:42:16,428 - INFO - Start persist model on server.
#     2025-07-20 21:42:16,431 - INFO - End persist model on server.
#     2025-07-20 21:42:16,431 - INFO - Finished FedAvg.



