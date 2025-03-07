# Edge Emulator

Edge emulator can be used to simulate multiple devices.

## Usage

The emulator can be started like this,

     python nvflare/edge/web/run_emulator.py config_json_file

## Configuration File

The `config_json_file` is a json file that defines the configuration of the emulator, with
following format:

```
{
  "endpoint": "http://localhost:4321",
  "num_devices": 16,
  "device_id_prefix": "emulator-",
  "processor": {
    "path": "sample_task_processor.SampleTaskProcessor",
    "args": {
      "data_file": "{device_id}/data/data-{user_id}.csv",
      "parameters": {
        "a": 123,
        "b": 456
      }
    }
  },
  "capabilities": {
    "methods": ["cnn"]
  }
}

```

The `endpoint`, `num_devices` and `capabilities` are self-explanatory. 

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
