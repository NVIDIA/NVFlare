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
  "processor": {
    "path": "sample_task_processor.SampleTaskProcessor",
    "python_path": "/opt/nvflare/edge/web",
    "args": {
      "data_file": "data/data.csv",
      "parameters": {
        "a": 123,
        "b": 456
      }
    }
  }
}

```

The `endpoint` and `num_devices` are self-explanatory. 

The `processor` defines the task processor that simulates the device's handling of tasks. This is a dictionary 
with following keys,
* `path`: The full path of the processor class
* `args`: A dictionary of arguments to be passed to the processor's constructor.
* `pypthon_path` defines where to find the processor module. If missing, the folder where the config file is will be used.
