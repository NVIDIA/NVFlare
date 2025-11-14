# Hello Pytorch Lightning
This example demonstrates how to use NVIDIA FLARE with PyTorch lightning to train an image classifier using federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-lightning/>`_. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation
for the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare
```

Get the example code from github: 
```
    git clone https://github.com/NVIDIA/NVFlare.git
```

then navigate to the hello-pt directory:

```
    git switch <release branch>
    cd examples/hello-world/hello-lightning
```

## Code Structure

```
    hello-lightning
    |
    |-- client.py        # client local training script
    |-- model.py         # model definition
    |-- job.py           # job recipe that defines client and server configurations
    |-- requirements.txt # dependencies
    |-- prepare_data.sh  # prepare data utils 
```

## Data
This example uses the CIFAR-10 dataset

In a real FL experiment, each client would have their own dataset used for their local training. You can download the CIFAR10 dataset from the Internet via torchvision's datasets module, You can split the datasets for different clients, so that each client has its own dataset. Here for simplicity's sake, the same dataset we will be using on each client.

The pytorch data module can download the datasets directly. since we have every site to download the same dataset, there are case, the training happens before the data is ready, which could lead to error. We can pre-download the data before we start the training by running from command line in a terminal

```
./prepare_data.sh
```

In PyTorch Lightning, a LightningDataModule is a standardized way to handle data loading and processing. It encapsulates all the steps required to prepare data for training, validation, and testing, making it easier to manage datasets and data loaders in a clean and organized manner. This abstraction helps separate data-related logic from the model and training code, promoting better code organization and reusability.

### LightningDataModule
* **Purpose**: The LightningDataModule is designed to encapsulate all data-related operations, including downloading, transforming, and splitting datasets, as well as providing data loaders for training, validation, testing, and prediction.
* **Key Methods**: - prepare_data(): Used for downloading and preparing data. This method is called only once and is not distributed across multiple GPUs or nodes. - setup(stage): Used to set up datasets for different stages (e.g., 'fit', 'validate', 'test', 'predict'). This method is called on every GPU or node. - train_dataloader(), val_dataloader(), test_dataloader(), predict_dataloader(): These methods return the respective data loaders for each stage.

### Setup of DataModule
In the CIFAR10DataModule, we have implemented the following:

* **Initialization (`__init__`)** : The constructor initializes the data directory and batch size, which are used throughout the data module.
* **Data Preparation (`prepare_data`)** : This method downloads the CIFAR-10 dataset if it is not already available in the specified directory. It prepares both the training and test datasets.
* **Setup (`setup`)**: This method assigns datasets for different stages: - For the 'fit' and 'validate' stages, it splits the CIFAR-10 training dataset into training and validation sets. - For the 'test' and 'predict' stages, it assigns the test dataset.
* **Data Loaders**: The module provides data loaders for training, validation, testing, and prediction, each configured with the specified batch size.
By using a LightningDataModule, the data handling logic is neatly encapsulated, making it easier to manage and modify data-related operations without affecting the rest of the training code.


## Model
In PyTorch Lightning, a LightningModule is a high-level abstraction built on top of PyTorch that streamlines the process of training models. It encapsulates the model architecture, training, validation, and testing logic, allowing developers to focus on the core components of their models without getting bogged down by the boilerplate code typically associated with PyTorch.

General Summary of a LightningModule

* **Model Definition**: The LightningModule is initialized with the model architecture, which is defined using PyTorch's nn.Module. This includes layers, activation functions, and any other components necessary for the model.
* **Forward Pass**: The forward method specifies how the input data flows through the model. This is where the core computation of the model is defined.
* **Training Logic**: The training_step method contains the logic for a single training iteration. It computes the loss and any metrics you wish to track, such as accuracy. This method is called automatically during the training loop.
* **Validation and Testing**: Similar to the training step, the validation_step and test_step methods define how the model is evaluated on validation and test datasets, respectively. These methods help in monitoring the model's performance and generalization.
* **Optimizer Configuration**: The configure_optimizers method specifies the optimizer(s) and learning rate scheduler(s) used during training. This allows for flexible and customizable training strategies.
By using a LightningModule, developers can leverage PyTorch Lightning's features like distributed training, automatic checkpointing, and logging, making it easier to scale experiments and manage complex training workflows. This abstraction promotes cleaner code, better organization, and easier debugging, ultimately accelerating the model development process.

 ```python
from typing import Any

import torch
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
```

## Client Code
Notice the training code is almost identical to the pytorch lightning standard training code. The only difference is that we added a few lines to receive and send data to the server. We mark all the changed code with number 0 to 4 to make it easier to understand.
 

The main flow of the code logic in the client.py file involves running a federated learning (FL) training logics locally on each client using PyTorch Lightning and NVFlare. Here's a breakdown of the key steps:

1 **Argument Parsing**: - The define_parser() function is used to parse command-line arguments, specifically the --batch_size argument, which sets the batch size for data loading.

2 **Initialization**: - The main() function begins by parsing the command-line arguments to get the batch size. - The flare.init() function is called to initialize the NVFlare client, which is necessary for using certain NVFlare functions like flare.get_site_name().

3 **Model and Data Module Setup**: - An instance of LitNet, a PyTorch Lightning model, is created. - An instance of CIFAR10DataModule is created with the specified batch size to handle data loading and processing.

4 **Trainer Configuration**: - A PyTorch Lightning Trainer is configured. If a GPU is available, it is set to use it; otherwise, it defaults to CPU.

5 **NVFlare Integration**: - The flare.patch(trainer) function is called to integrate NVFlare with the PyTorch Lightning trainer. This allows the trainer to handle federated learning tasks.

6 **Federated Learning Loop**: - A loop runs while flare.is_running() returns True, indicating that the federated learning job is active.
 
  Within the loop:

  > * The global model is received from the NVFlare server using flare.receive().
  > * The current round and site name are printed for logging purposes.
  > * The global model is validated using trainer.validate().
  > * Local training is performed using trainer.fit(), starting with the received global model.
  > * The local model is tested using trainer.test().
  > * Predictions are made using trainer.predict().
  
7 **Execution**: - The main() function is executed if the script is run as the main module, starting the entire process.


With this method, the developers can use the Client API
to change their centralized training code to an FL scenario with
these simple code changes shown below.
```
    # (1) import nvflare lightning client API
    import nvflare.client.lightning as flare
    
    # (2) patch the lightning trainer
    flare.patch(trainer)
    
    while flare.is_running():
        # Note that we can optionally receive the FLModel from NVFLARE.
        # We don't need to pass this input_model to trainer because after flare.patch 
        # the trainer.fit/validate will get the global model internally
        input_model = flare.receive()
    
        trainer.validate(...)
    
        trainer.fit(...)
    
        trainer.test(...)
    
        trainer.predict(...)
```


## Server Code
In federated averaging, the server code is responsible for aggregating model updates from clients, the workflow pattern is similar to scatter-gather. 


First, we provide a simple implementation of the [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com) algorithm with NVFlare. 
The `run()` routine implements the main algorithmic logic. 
Subroutines, like `sample_clients()` and `scatter_and_gather_model()` utilize the communicator object, native to each Controller to get the list of available clients,
distribute the current global model to the clients, and collect their results.

The FedAvg controller implements these main steps:
1. FL server initializes an initial model using `self.load_model()`.
2. For each round (global iteration):
    - FL server samples available clients using `self.sample_clients()`.
    - FL server sends the global model to clients and waits for their updates using `self.send_model_and_wait()`.
    - FL server aggregates all the `results` and produces a new global model using `self.update_model()`.

```python
class FedAvg(BaseFedAvg):
    def run(self) -> None:
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(results)

            model = self.update_model(model, aggregate_results)

            self.save_model(model)

        self.info("Finished FedAvg.")
```

In this example, we will directly use the default federated averaging algorithm provided by NVFlare. The FedAvg class is defined in nvflare.app_common.workflows.fedavg.FedAvg There is no need to defined a customized server code for this example.


## Job Recipe Code
The job recipe code is used to define the client and server configurations.

## Run FL Job
This section provides the command to execute the federated learning job using the job recipe defined above. Run this command in your terminal. First, run the following command to download the data:

```
./prepare_data.sh
```

## Command to execute the FL job

Use the following command in your terminal to start the job with the specified number of rounds, batch size, and number of clients.
```
python job.py --num_rounds 2 --batch_size 16
```

output

```
        # < ... skip few lines of logs ..>
        # 2025-07-22 18:45:45,758 - INFO - Start FedAvg.
        # 2025-07-22 18:45:45,759 - INFO - loading initial model from persistor
        # 2025-07-22 18:45:45,759 - INFO - Both source_ckpt_file_full_name and ckpt_preload_path are not provided. Using the default model weights initialized on the persistor side.
        # 2025-07-22 18:45:45,760 - INFO - Round 0 started.
        # 2025-07-22 18:45:45,760 - INFO - Sampled clients: ['site-1', 'site-2']
        # 2025-07-22 18:45:45,760 - INFO - Sending task train to ['site-1', 'site-2']
        #
        # < ... skip .. few lines of logs ..>
        #
        # 2025-07-22 18:45:50,507 - INFO - batch_size=16, site=site-1
        # 2025-07-22 18:45:50,543 - INFO -
        # [Current Round=0, Site = site-1]
        #
        # 2025-07-22 18:45:50,543 - INFO - --- validate global model ---
        # 2025-07-22 18:45:50,578 - INFO - batch_size=16, site=site-2
        # 2025-07-22 18:45:50,656 - INFO -
        # [Current Round=0, Site = site-2]
        #
        # 2025-07-22 18:45:50,656 - INFO - --- validate global model ---
        #
        # < ... skip .. few lines of logs ..>
        #
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃        Test metric        ┃       DataLoader 0        ┃
        # ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        # │      test_acc_epoch       │    0.44699999690055847    │
        # │         test_loss         │    1.5125484466552734     │
        # └───────────────────────────┴───────────────────────────┘
        # Testing DataLoader 0:  68%|████████████████████████████████▍               | 422/625 [00:01<00:00, 276.33it/s]2025-07-22 18:46:39,629 - INFO - --- prediction with new best model ---
        # Testing DataLoader 0:  76%|████████████████████████████████████▋           | 478/625 [00:01<00:00, 275.61it/s]2025-07-22 18:46:39,837 - INFO - Files already downloaded and verified
        # Testing DataLoader 0: 100%|████████████████████████████████████████████████| 625/625 [00:02<00:00, 275.79it/s]
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃        Test metric        ┃       DataLoader 0        ┃
        # ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        # │      test_acc_epoch       │    0.44699999690055847    │
        # │         test_loss         │    1.5125484466552734     │
        # └───────────────────────────┴───────────────────────────┘
        # 2025-07-22 18:46:40,370 - INFO - --- prediction with new best model ---
        # 2025-07-22 18:46:40,431 - INFO - Files already downloaded and verified
        # 2025-07-22 18:46:40,577 - INFO - Files already downloaded and verified
        # Predicting DataLoader 0:  16%|███████▍                                     | 103/625 [00:00<00:01, 371.90it/s]2025-07-22 18:46:41,191 - INFO - Files already downloaded and verified
        # Predicting DataLoader 0: 100%|█████████████████████████████████████████████| 625/625 [00:01<00:00, 367.54it/s]
        # Predicting DataLoader 0:  53%|███████████████████████▊                     | 331/625 [00:00<00:00, 346.29it/s]2025-07-22 18:46:42,615 - WARNING - request to stop the job for reason END_RUN received
        # Predicting DataLoader 0: 100%|█████████████████████████████████████████████| 625/625 [00:01<00:00, 344.12it/s]
        # 2025-07-22 18:46:43,476 - WARNING - request to stop the job for reason END_RUN received
```