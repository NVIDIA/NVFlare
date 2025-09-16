# DeviceSimulator for FL Pipeline Prototyping

The DeviceSimulator can be used to simulate multiple devices locally to test a cross-device FL pipeline.

## Usage

The DeviceSimulator can be started like this,

     python run_device_simulator.py config_json_file

## Configuration File

The `config_json_file` is a json file that defines the configuration of the DeviceSimulator, with
following format:

```
{
  "endpoint": "http://localhost:4321",
  "num_devices": 1000,
  "num_workers": 50,
  "processor": {
    "path": "sample_task_processor.SampleTaskProcessor",
    "args": {
      "data_file": "{device_id}/data/data-{user_id}.csv",
      "parameters": {
        "a": 123,
        "b": 456
      }
    }
  }
}

```

The `endpoint` is self-explanatory. 

The `num_devices` specifies the total number of devices to be simulated.

The `processor` defines the task processor that simulates the device's handling of tasks. This is a dictionary 
with following keys,
* `path`: The full path of the processor class
* `args`: A dictionary of arguments to be passed to the processor's constructor.
* `pypthon_path` defines where to find the processor module. If missing, the folder where the config file is will be used.

## Variable Substitution

The args of the processor can contain variables. They will be substituted with the device specific information.

Variable are enclosed with curly braces, like `{device_id}`.

Currently, only following variables are supported,

* `{device_id}`: the device ID
* `{user_id}`: the user ID

This can be used to support different data files for different devices, like

```{"data_file": "{device_id}/data/data-{user_id}.csv"}```
