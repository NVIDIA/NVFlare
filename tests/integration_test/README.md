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

| Attributes          | Description                                                 |
|---------------------|-------------------------------------------------------------|
| `n_servers`         | Number of servers                                           |
| `n_clients`         | Number of clients                                           |
| `single_app_as_job` | Whether the test cases are single app job or not.           |
| `cleanup`           | Whether to clean up test folders or not. (Default to True.) |
| `tests`             | The test cases to run                                       |
| `ha`                | Whether to use provision mode or not. (Default to False.)   |

If `single_app_as_job` is True, `apps_root_dir` is required.
Otherwise, `jobs_root_dir` is required.


### Test cases

Each test case has the following attributes:

| Attributes           | Description                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------|
| app_name or job_name | testing app or job folder name. Note that these folders need to be inside the jobs_root_dir. |
| validators           | Which validator to use to validate the running result once the job is finished.              |
| event_sequence_yaml  | What event sequence to run during each jobs. (HA test cases)                                 |

### Event sequence

Each event sequence has the following attributes:

| Attributes  | Description                                  |
|-------------|----------------------------------------------|
| description | Description of this specific event sequence. |
| events      | A list of events                             |

Each event has the following attributes:

| Attributes   | Description                                            |
|--------------|--------------------------------------------------------|
| trigger      | When to perform the specified action.                  |
| action       | What actions to take if the trigger is triggered.      |
| result_state | What state to expect after these actions are finished. |

Triggers can be triggered based on its type:
  - str: match a string based on log output from server
  - dict: state based on predefined tracked state variables
    (workflow, task, round_number, run_finished etc.)

## Folder Structure

- src: source codes for integration test system
- data:
  - apps: applications for testing
  - event_sequence: ha test event sequence yaml files
  - single_app_as_job: test configuration for single app as job test cases
- tf2: TensorFlow2 related codes for the applications used in
  integration tests.
- validators: Codes that implement the logic to validate the running result
  once the job is finished.

### apps

Because the applications inside the `apps` folder is treated as single app job.

Each application in apps folder should contain `config` folder.

And should have `config_fed_server.json` and `config_fed_client.json` inside the config folder.
