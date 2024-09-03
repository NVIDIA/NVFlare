# Hello FedAvg

In this example we highlight the flexibility of the ModelController API, and show how to write a Federated Averaging workflow with early stopping, model selection, and saving and loading. Follow along in the [hello-fedavg.ipynb](hello-fedavg.ipynb) notebook for more details.

### 1. Setup

```
pip install nvflare~=2.5.0rc torch torchvision tensorboard
```

Download the dataset:
```
./prepare_data.sh
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
