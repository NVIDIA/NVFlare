"""
`Hello Pytorch <../hello-lightning/doc.html>`_||
**hello lightning** ||
`hello tensorflow <../hello-tf/doc.html>`_ ||
`hello LR <../hello-lr/doc.html>`_ ||
`hello KMeans <../hello-kmeans/doc.html>`_ ||
`hello KM <../hello-km/doc.html>`_ ||
`hello stats <../hello-stats/doc.html>`_ ||
`hello cyclic <../hello-cyclic/doc.html>`_ ||
`hello-xgboost <../hello-xgboost/doc.html>`_ ||
`hello-flower <../hello-flower/doc.html>`_ ||


Hello Pytorch Lightning
======================

This example demonstrates how to use NVIDIA FLARE with PyTorch lightning to train an image classifier using
federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-lightning/>`.
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
 hello-lightning
    |
    |-- client.py        # client local training script
    |-- model.py         # model definition
    |-- job.py              # job recipe that defines client and server configurations
    |-- requirements.txt    # dependencies

Data
-----------------
This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module,
You can split the datasets for different clients, so that each client has its own dataset.
Here for simplicity's sake, the same dataset we will be using on each client.

In PyTorch Lightning, a `LightningDataModule` is a standardized way to handle data loading and processing. It encapsulates all the steps required to prepare data for training, validation, and testing, making it easier to manage datasets and data loaders in a clean and organized manner. This abstraction helps separate data-related logic from the model and training code, promoting better code organization and reusability.

### Brief Description of `LightningDataModule`

- **Purpose:** The `LightningDataModule` is designed to encapsulate all data-related operations, including downloading, transforming, and splitting datasets, as well as providing data loaders for training, validation, testing, and prediction.

- **Key Methods:**
  - `prepare_data()`: Used for downloading and preparing data. This method is called only once and is not distributed across multiple GPUs or nodes.
  - `setup(stage)`: Used to set up datasets for different stages (e.g., 'fit', 'validate', 'test', 'predict'). This method is called on every GPU or node.
  - `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `predict_dataloader()`: These methods return the respective data loaders for each stage.

### Setup of `CIFAR10DataModule`

In the `CIFAR10DataModule`, we have implemented the following:

- **Initialization (`__init__`):** The constructor initializes the data directory and batch size, which are used throughout the data module.

- **Data Preparation (`prepare_data`):** This method downloads the CIFAR-10 dataset if it is not already available in the specified directory. It prepares both the training and test datasets.

- **Setup (`setup`):** This method assigns datasets for different stages:
  - For the 'fit' and 'validate' stages, it splits the CIFAR-10 training dataset into training and validation sets.
  - For the 'test' and 'predict' stages, it assigns the test dataset.

- **Data Loaders:** The module provides data loaders for training, validation, testing, and prediction, each configured with the specified batch size.

By using a `LightningDataModule`, the data handling logic is neatly encapsulated, making it easier to manage and modify data-related operations without affecting the rest of the training code.

"""
class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = DATASET_PATH, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "validate":
            cifar_full = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=False, transform=transform
            )
            self.cifar_train, self.cifar_val = random_split(cifar_full, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.cifar_test = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=False, transform=transform
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)


################################
# Model
# ------------------
# In PyTorch Lightning, a `LightningModule` is a high-level abstraction built
# on top of PyTorch that streamlines the process of training models. It
# encapsulates the model architecture, training, validation, and testing logic,
# allowing developers to focus on the core components of their models without
# getting bogged down by the boilerplate code typically associated with PyTorch.
#
# General Summary of a `LightningModule`
#
# - **Model Definition:** The `LightningModule` is initialized with the model
#   architecture, which is defined using PyTorch's `nn.Module`. This includes
#   layers, activation functions, and any other components necessary for the
#   model.
#
# - **Forward Pass:** The `forward` method specifies how the input data flows
#   through the model. This is where the core computation of the model is
#   defined.
#
# - **Training Logic:** The `training_step` method contains the logic for a
#   single training iteration. It computes the loss and any metrics you wish to
#   track, such as accuracy. This method is called automatically during the
#   training loop.
#
# - **Validation and Testing:** Similar to the training step, the
#   `validation_step` and `test_step` methods define how the model is evaluated
#   on validation and test datasets, respectively. These methods help in
#   monitoring the model's performance and generalization.
#
# - **Optimizer Configuration:** The `configure_optimizers` method specifies the
#   optimizer(s) and learning rate scheduler(s) used during training. This
#   allows for flexible and customizable training strategies.
#
# By using a `LightningModule`, developers can leverage PyTorch Lightning's
# features like distributed training, automatic checkpointing, and logging,
# making it easier to scale experiments and manage complex training workflows.
# This abstraction promotes cleaner code, better organization, and easier
# debugging, ultimately accelerating the model development process.


from typing import Any

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

NUM_CLASSES = 10
criterion = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
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


class LitNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        # (optional) pass additional information via self.__fl_meta__
        self.__fl_meta__ = {}

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        return loss

    def evaluate(self, batch, stage=None):
        x, labels = batch
        outputs = self(x)
        loss = criterion(outputs, labels)
        self.valid_acc(outputs, labels)

        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", self.valid_acc, on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.evaluate(batch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return {"optimizer": optimizer}


######################################################################
# --------------
#


#####################################################################
# Client Code
# ------------------
#
# The client code is responsible for
# Notice the training code is almost identical to the pytorch lightning standard training code.
# The only difference is that we added a few lines to receive and send data to the server.
# We mark all the chagned code with number 0 to 4 to make it easier to understand.
#
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from model import LitNet
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from torch.utils.data import DataLoader, random_split

# (0) import nvflare lightning client API
import nvflare.client.lightning as flare

seed_everything(7)


DATASET_PATH = "/tmp/nvflare/data"
BATCH_SIZE = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CIFAR10DataModule(LightningDataModule):
       # <skip rest of code> 
       # described in data section above
       pass



def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)

    return parser.parse_args()


def main():
    args = define_parser()
    batch_size = args.batch_size

    # (1) flare.init() is only needed if the flare function is used (such as flare.get_site_name())
    flare.init()
    print(f"batch_size={batch_size}, site={flare.get_site_name()}")

    model = LitNet()
    cifar10_dm = CIFAR10DataModule(batch_size=batch_size)
    if torch.cuda.is_available():
        trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1 if torch.cuda.is_available() else None)
    else:
        trainer = Trainer(max_epochs=1, devices=None)

    # (2) patch the lightning trainer
    flare.patch(trainer)

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        # Note that we don't need to pass this input_model to trainer
        # because after flare.patch the trainer.fit/validate will get the
        # global model internally
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

        # (4) evaluate the current global model to allow server-side model selection
        print("--- validate global model ---")
        trainer.validate(model, datamodule=cifar10_dm)

        # perform local training starting with the received global model
        print("--- train new model ---")
        trainer.fit(model, datamodule=cifar10_dm)

        # test local model
        print("--- test new model ---")
        trainer.test(ckpt_path="best", datamodule=cifar10_dm)

        # get predictions
        print("--- prediction with new best model ---")
        trainer.predict(ckpt_path="best", datamodule=cifar10_dm)


if __name__ == "__main__":
    main()

# The main flow of the code logic in the `client.py` file involves running a federated learning (FL) training logics locally on each client using PyTorch Lightning and NVFlare. 
# Here's a breakdown of the key steps:

# 1. **Argument Parsing:**
#    - The `define_parser()` function is used to parse command-line arguments, specifically the `--batch_size` argument, which sets the batch size for data loading.

# 2. **Initialization:**
#    - The `main()` function begins by parsing the command-line arguments to get the batch size.
#    - The `flare.init()` function is called to initialize the NVFlare client, which is necessary for using certain NVFlare functions like `flare.get_site_name()`.

# 3. **Model and Data Module Setup:**
#    - An instance of `LitNet`, a PyTorch Lightning model, is created.
#    - An instance of `CIFAR10DataModule` is created with the specified batch size to handle data loading and processing.

# 4. **Trainer Configuration:**
#    - A PyTorch Lightning `Trainer` is configured. If a GPU is available, it is set to use it; otherwise, it defaults to CPU.

# 5. **NVFlare Integration:**
#    - The `flare.patch(trainer)` function is called to integrate NVFlare with the PyTorch Lightning trainer. This allows the trainer to handle federated learning tasks.

# 6. **Federated Learning Loop:**
#    - A loop runs while `flare.is_running()` returns `True`, indicating that the federated learning job is active.
#    - Within the loop:
#      - The global model is received from the NVFlare server using `flare.receive()`.
#      - The current round and site name are printed for logging purposes.
#      - The global model is validated using `trainer.validate()`.
#      - Local training is performed using `trainer.fit()`, starting with the received global model.
#      - The local model is tested using `trainer.test()`.
#      - Predictions are made using `trainer.predict()`.

# 7. **Execution:**
#    - The `main()` function is executed if the script is run as the main module, starting the entire process.



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
# The job recipe code is used to define the client and server configurations.


import argparse

from model import LitNet
from nvflare.app_opt.pt.job_config.Job_recipe import FedAvgRecipe


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    batch_size = args.batch_size

    recipe = FedAvgRecipe(clients=n_clients,
                          num_rounds=num_rounds,
                          model= LitNet(),
                          client_script="client.py",
                          client_script_args= f"--batch_size {batch_size}")

    recipe.execute()



if __name__ == "__main__":
    main()


#####################################################################
# Run FL Job
# ------------------
#
# This section provides the command to execute the federated learning job
# using the job recipe defined above. Run this command in your terminal.

# Command to execute the FL job
# -----------------------------
# Use the following command in your terminal to start the job with the specified
# number of rounds, batch size, and number of clients.

# python job.py --num_rounds 2 --batch_size 16 


