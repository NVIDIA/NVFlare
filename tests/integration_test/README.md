# NVIDIA FLARE Integration test

The integration tests entry file is `tests/integration_test/system_test.py`.
The overseer tests entry file is `tests/integration_test/overseer_test.py`

## Setup

Note that the HA test cases are using `./data/project.yml` to provision the whole system.
That will require we have `localhost0` and `localhost1` map to `127.0.0.1`.
You need to either modify `/etc/hosts` file before running the test.
Or if you are running using docker container you should use `--add-host localhost0:127.0.0.1`.

## Run

First switch to this folder and then run

`PYTHONPATH=[path/to/your/NVFlare] ./run_integration_tests.sh`


## Test structure

The main test will read test configurations from `./test_cases.yml`.

### Test config

Each test configuration yaml should contain the following attributes:

| Attributes        | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `n_servers`       | Number of servers                                           |
| `n_clients`       | Number of clients                                           |
| `cleanup`         | Whether to clean up test folders or not. (Default to True.) |
| `ha`              | Whether to use provision mode or not. (Default to False.)   |
| `tests`           | The test cases to run                                       |
| `jobs_root_dir`   | The directory that contains the job folders to upload       |

### Test cases

Each test case has the following attributes:

| Attributes            | Description                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------|
| test_name             | Name of this test.                                                                         |
| event_sequence        | What events to run for each test.                                                          |
| validators (optional) | Which validator to use to validate the running result once the event sequence is finished. |

### Event sequence

Event sequence is consist of a list of events. 

Each event has the following attributes:

| Attributes | Description                                             |
|------------|---------------------------------------------------------|
| trigger    | When to perform the specified action.                   |
| actions    | What actions to take if the trigger is triggered.       |
| result     | What result to expect after these actions are finished. |

Each trigger has the following attributes:

| Attributes   | Description                       |
|--------------|-----------------------------------|
| type         | Specify the type of this trigger. |
| data         | The data of this trigger.         |

The following trigger type is supported:
  - server_log: triggered if the data can be found in server's log
  - client_log: triggered if the data can be found in client's log
  - server_job_log: triggered if the data can be found in server's job's log
  - client_job_log: triggered if the data can be found in client's job's log
  - run_state: triggered if the run state matched the data

Note that:

   `run_state` is based on predefined tracked state variables
    (workflow, task, round_number, run_finished etc.)

Each result has the following attributes:

| Attributes   | Description                      |
|--------------|----------------------------------|
| type         | Specify the type of this result. |
| data         | The data of this result.         |

The following result type is supported:
  - run_state: check if current `run_state` match data.
  - admin_api_response: check if admin_api_response match data.

## Folder Structure

- src: source codes for the integration test system
- data:
  - apps: applications for testing
  - test configurations
- tf2: TensorFlow2 related codes for the applications used in integration tests
- validators: Codes that implement the logic to validate the running result
  once the job is finished

### apps

Each application in apps folder should be a valid NVFlare application.
