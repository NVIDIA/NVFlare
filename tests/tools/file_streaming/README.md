# Streaming API Example Jobs

The `file_streaming` is a NVFlare job that demonstrates how to use FileStreamer to send large 
files without memory limitation.

It can also be used to benchmark/test file transferring performance of NVFlare FileStreamer API.

## Configuration Files

The enclosed configuration files should be modified before the job is run.

### Server Configuration

`app/config/config_fed_server.json`:

```
{
  "format_version": 2,
  "task_data_filters": [],
  "task_result_filters": [],
  "workflows": [
    {
      "id": "controller",
      "path": "controller.FileTransferController",
      "args": {}
    }
  ],
  "components": [
    {
      "id": "receiver",
      "path": "file_streaming.FileReceiver",
      "args": {
        "output_folder": "/tmp"
      }
    }
  ]
}
```

The `output_folder` points to where the received file will be stored.

Please be careful, if a file with the same name as transferred file exists in the folder, 
the existing file will be overwritten.

### Client Configuration

`app/config/config_fed_client.json`:

```
{
  "format_version": 2,
  "executors": [
    {
      "tasks": [
        "train"
      ],
      "executor": {
        "path": "executor.DummyExecutor",
        "args": {}
      }
    }
  ],
  "task_result_filters": [],
  "task_data_filters": [],
  "components": [
    {
      "id": "sender",
      "path": "file_streaming.FileSender",
      "args": {
        "file_name": "/tmp/data/test.data"
      }
    }
  ]
}

```

The `file_name` must point to an existing file to be sent.

The workflow doesn't require an executor but NVFlare job must have an executor configured so a mock one is used.

## Running the Job

### Simulator 

To test locally on one machine, the job can run in simulator like this,

```commandline
nvflare simulator -n 1 -t 1 -l full tests/tools/file_streaming
```

Please add `-l full` to the simulator command. Otherwise, the relevant logs are not printed.

### POC

To test on different machines over network, NVFlare POC mode should be used. Here are the steps,

1. Provision the POC system as usual
2. Start server on the receiving machine 
3. Start client on the sending machine
4. Submit the job as usual

Following log line should show up on the server after file transferring is complete,

    2025-03-26 15:40:14,532 - FileReceiver - INFO - [identity=simulator_server, run=simulate_job, wf=controller, peer=site-1, peer_run=simulate_job] - File /tmp/test.data with 2117698317 bytes received in 7.358 seconds
