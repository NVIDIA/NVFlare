
Hello Pytorch Lightning
=======================

This example demonstrates how to use NVIDIA FLARE with PyTorch lightning to train an image classifier using
federated averaging (FedAvg).The complete example code can be found in the`hello-lightning directory <examples/hello-world/hello-lightning/>`_.
It is recommended to create a virtual environment and run everything within a virtualenv.


NVIDIA FLARE Installation
-------------------------

for the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare


get the example code from github:

.. code-block:: text

   git clone https://github.com/NVIDIA/NVFlare.git

then navigate to the hello-lightning directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-lightning

Install the dependency

.. code-block:: bash

   pip install -r requirements.txt

Code Structure
--------------

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
This example uses the `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset

In a real FL experiment, each client would have their own dataset used for their local training.
You can download the CIFAR10 dataset from the Internet via torchvision's datasets module,
You can split the datasets for different clients, so that each client has its own dataset.
Here for simplicity's sake, the same dataset we will be using on each client.

The pytorch data module can download the datasets directly. since we have every site to download the same dataset,
there are case, the training happens before the data is ready, which could lead to error. We can pre-download the data
before we start the training by running from command line in a terminal

.. code-block:: text

    ./prepare_data.sh


.. literalinclude:: ../../../examples/hello-world/hello-lightning/prepare_data.sh
    :language: bash
    :linenos:
    :caption: prepare_data.sh

In PyTorch Lightning, a `LightningDataModule` is a standardized way to handle data loading and processing. It encapsulates all the steps required to prepare data for training, validation, and testing, making it easier to manage datasets and data loaders in a clean and organized manner. This abstraction helps separate data-related logic from the model and training code, promoting better code organization and reusability.

`LightningDataModule`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Purpose:** The `LightningDataModule` is designed to encapsulate all data-related operations, including downloading, transforming, and splitting datasets, as well as providing data loaders for training, validation, testing, and prediction.

- **Key Methods:**
  - `prepare_data()`: Used for downloading and preparing data. This method is called only once and is not distributed across multiple GPUs or nodes.
  - `setup(stage)`: Used to set up datasets for different stages (e.g., 'fit', 'validate', 'test', 'predict'). This method is called on every GPU or node.
  - `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `predict_dataloader()`: These methods return the respective data loaders for each stage.

Setup of `DataModule`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the `CIFAR10DataModule`, we have implemented the following:

- **Initialization (`__init__`):** The constructor initializes the data directory and batch size, which are used throughout the data module.

- **Data Preparation (`prepare_data`):** This method downloads the CIFAR-10 dataset if it is not already available in the specified directory. It prepares both the training and test datasets.

- **Setup (`setup`):** This method assigns datasets for different stages:
  - For the 'fit' and 'validate' stages, it splits the CIFAR-10 training dataset into training and validation sets.
  - For the 'test' and 'predict' stages, it assigns the test dataset.

- **Data Loaders:** The module provides data loaders for training, validation, testing, and prediction, each configured with the specified batch size.

By using a `LightningDataModule`, the data handling logic is neatly encapsulated, making it easier to manage and modify data-related operations without affecting the rest of the training code.

.. literalinclude:: ../../../examples/hello-world/hello-lightning/client.py
    :language: python
    :linenos:
    :caption: data module
    :lines: 14-70


Model
------------------
In PyTorch Lightning, a `LightningModule` is a high-level abstraction built
on top of PyTorch that streamlines the process of training models. It
encapsulates the model architecture, training, validation, and testing logic,
allowing developers to focus on the core components of their models without
getting bogged down by the boilerplate code typically associated with PyTorch.

General Summary of a `LightningModule`

- **Model Definition:** The `LightningModule` is initialized with the model
  architecture, which is defined using PyTorch's `nn.Module`. This includes
  layers, activation functions, and any other components necessary for the
  model.

- **Forward Pass:** The `forward` method specifies how the input data flows
  through the model. This is where the core computation of the model is
  defined.

- **Training Logic:** The `training_step` method contains the logic for a
  single training iteration. It computes the loss and any metrics you wish to
  track, such as accuracy. This method is called automatically during the
  training loop.

- **Validation and Testing:** Similar to the training step, the
  `validation_step` and `test_step` methods define how the model is evaluated
  on validation and test datasets, respectively. These methods help in
  monitoring the model's performance and generalization.

- **Optimizer Configuration:** The `configure_optimizers` method specifies the
  optimizer(s) and learning rate scheduler(s) used during training. This
  allows for flexible and customizable training strategies.

By using a `LightningModule`, developers can leverage PyTorch Lightning's
features like distributed training, automatic checkpointing, and logging,
making it easier to scale experiments and manage complex training workflows.
This abstraction promotes cleaner code, better organization, and easier
debugging, ultimately accelerating the model development process.

.. literalinclude:: ../../../examples/hello-world/hello-lightning/model.py
    :language: python
    :linenos:
    :caption: model.py
    :lines: 14-

--------------


Client Code
------------------

Notice the training code is almost identical to the pytorch lightning standard training code.
The only difference is that we added a few lines to receive and send data to the server.
We mark all the changed code with number 0 to 4 to make it easier to understand.


.. literalinclude:: ../../../examples/hello-world/hello-lightning/client.py
    :language: python
    :linenos:
    :caption: client.py
    :lines: 71-


The main flow of the code logic in the `client.py` file involves running a federated learning (FL) training logics locally on each client using PyTorch Lightning and NVFlare. 
Here's a breakdown of the key steps:

1. **Argument Parsing:**

   - The `define_parser()` function is used to parse command-line arguments, specifically the `--batch_size` argument, which sets the batch size for data loading.

2. **Initialization:**

   - The `main()` function begins by parsing the command-line arguments to get the batch size.
   - The `flare.init()` function is called to initialize the NVFlare client, which is necessary for using certain NVFlare functions like `flare.get_site_name()`.

3. **Model and Data Module Setup:**

   - An instance of `LitNet`, a PyTorch Lightning model, is created.
   - An instance of `CIFAR10DataModule` is created with the specified batch size to handle data loading and processing.

4. **Trainer Configuration:**

   - A PyTorch Lightning `Trainer` is configured. If a GPU is available, it is set to use it; otherwise, it defaults to CPU.

5. **NVFlare Integration:**

   - The `flare.patch(trainer)` function is called to integrate NVFlare with the PyTorch Lightning trainer. This allows the trainer to handle federated learning tasks.

6. **Federated Learning Loop:**

   - A loop runs while `flare.is_running()` returns `True`, indicating that the federated learning job is active.
   - Within the loop:
      - The global model is received from the NVFlare server using `flare.receive()`.
      - The current round and site name are printed for logging purposes.
      - The global model is validated using `trainer.validate()`.
      - Local training is performed using `trainer.fit()`, starting with the received global model.
      - The local model is tested using `trainer.test()`.
      - Predictions are made using `trainer.predict()`.

7. **Execution:**

   - The `main()` function is executed if the script is run as the main module, starting the entire process.


Server Code
------------------
In federated averaging, the server code is responsible for
aggregating model updates from clients, the workflow pattern is similar to scatter-gather.
In this example, we will directly use the default federated averaging algorithm provided by NVFlare.
The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
There is no need to defined a customized server code for this example.


Job Recipe Code
------------------
The job recipe code is used to define the client and server configurations.

.. literalinclude:: ../../../examples/hello-world/hello-lightning/job.py
    :language: python
    :linenos:
    :caption: Job Recipe (job.py)
    :lines: 14-


Run FL Job
------------------

This section provides the command to execute the federated learning job
using the job recipe defined above. Run this command in your terminal.
First, run the following command to download the data:

.. code-block:: text

  ./prepare_data.sh


**Command to execute the FL job**

Use the following command in your terminal to start the job with the specified
number of rounds, batch size, and number of clients.


.. code-block:: text

  python job.py --num_rounds 2 --batch_size 16


output

.. code-block:: text


.. code-block:: Python
   :dedent: 1

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