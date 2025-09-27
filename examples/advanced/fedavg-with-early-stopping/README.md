# Hello FedAvg With Early Stopping

In this example we highlight the flexibility of the ModelController API, and show how to write a Federated Averaging workflow with early stopping, model selection, and saving and loading. Follow along in the [hello-fedavg.ipynb](hello-fedavg.ipynb) notebook for more details.

## NVIDIA FLARE Installation
for the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare

```
Install the dependency

```
pip install -r requirements.txt
```
## Code Structure
first get the example code from github:

```
git clone https://github.com/NVIDIA/NVFlare.git
```
then navigate to the hello-custom-fedavg directory:

```
git switch <release branch>
cd examples/hello-world/hello-custom-fedavg
```
``` bash
hello-pt
|
|-- client.py         # client local training script
|-- model.py          # model definition
|-- job.py            # job recipe that defines client and server configurations
|-- prepare_data.sh   # download the CIFAR10 data script 
|-- requirements.txt  # dependencies
```
Download the dataset:
```
./prepare_data.sh
```

### 2. PTFedAvgEarlyStopping using ModelController API

The ModelController API enables the option to easily customize a workflow with Python code.

- FedAvg: We subclass the BaseFedAvg class to leverage the predefined aggregation functions.
- Early Stopping: We add a `stop_condition` argument (eg. `"accuracy >= 80"`) and end the workflow early if the corresponding global model metric meets the condition.
- Patience: If set to a value greater than 1, the FL experiment will stop if the defined `stop_condition` does not improve over X consecutive FL rounds.
- Task to optimize: Allows the user to specify which task to apply the `early stopping` mechanism to (e.g., the validation phase)
- Model Selection: As and alternative to using a `IntimeModelSelector` componenet for model selection, we instead compare the metrics of the models in the workflow to select the best model each round.
- Saving/Loading: Rather than configuring a persistor such as `PTFileModelPersistor` component, we choose to utilize PyTorch's save and load functions and save the metadata of the FLModel separately.

### 3. Run the Job 

Use the Job API to define and run the example with the simulator:

First download the data
```bash
./prepare_data.sh
```

```bash
python job.py
``` 