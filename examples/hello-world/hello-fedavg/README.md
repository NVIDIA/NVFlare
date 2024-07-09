# Hello FedAvg

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework.
In this example we highlight the flexibility of the ModelController API, and show how to write a Federated Averaging workflow with early stopping, model selection, and saving and loading.

### 1. Setup

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

Install additional requirements:

```
pip3 install -r requirements.txt
```


Prepare the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:

```
bash ./prepare_data.sh
```

### 2. PTFedAvgEarlyStopping using ModelController API

The ModelController API enables the option to easily customize a workflow with Python code.

- FedAvg: We subclass the BaseFedAvg class to leverage the predefined aggregation functions.
- Early Stopping: We add a `stop_condition` argument (eg. `"accuracy >= 80"`) and end the workflow early if the corresponding global model metric meets the condition.
- Model Selection: As and alternative to using a `IntimeModelSelector` componenet for model selection, we instead compare the metrics of the models in the workflow to select the best model each round.
- Saving/Loading: Rather than configuring a persistor such as `PTFileModelPersistor` component, we choose to utilize PyTorch's save and load functions and save the metadata of the FLModel separately.

### 3. Run the script

Use the Job API to define and run the example with the simulator:

```
python3 pt_fedavg_early_stopping_script.py
```

View the results in the job workspace: `/tmp/nvflare/jobs/workdir`.
