Hello FedAvg Metric Optimization
================================

In this example we will introduce the FedAvg Metric Optimization. The
FedAvgMetricOptimization class combines two techniques—early stopping and metric
optimization (minimization or maximization)—to identify the best model during FL
learning. It uses a patience parameter, which specifies how many FL rounds to wait
without any metric improvement before stopping the training.

1. Setup

```cmd
pip install nvflare torch torchvision tensorboard
```

2. PTFedAvgMetricOptimization using ModelController API

The ModelController API enables the option to easily customize a workflow with Python code.

* FedAvg: We subclass the BaseFedAvg class to leverage the predefined aggregation functions.
* Metric Optimization: We have added a set of new arguments, most of which are optional, to provide greater flexibility in configuring the metric optimization:
    - `target_metric_name`: specifies the name of the metric to optimize
    - `optimization mode`: choose whether to minimize or maximize the metric
    - `task_train_name`: specifies the name of the training task
    - `task_validation_name`: specifies the name of the validation task
    - `task_to_optimize`: indicates whether to apply metric optimization to the training or validation task
    - `patience`: defines the number of FL rounds to wait without improvement before stopping the training
* Model Selection: As and alternative to using a IntimeModelSelector componenet for model selection, we instead compare the metrics of the models in the workflow to select the best model each round.
* Saving/Loading: Rather than configuring a persistor such as PTFileModelPersistor component, we choose to utilize PyTorch's save and load functions and save the metadata of the FLModel separately.

3. Run the script

Use the Job API to define and run the example with the simulator:

```python
python pt_fedavg_metric_optimization_script.py
```
