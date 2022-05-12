# NVIDIA FLARE Integration test

The integration tests entry file is `test/integration_test/system_test.py`.

## Run

First switch to this folder and then run

`PYTHONPATH=[path/to/your/NVFlare] ./run_integration_tests.sh`


## Test structure

The main test will read test configurations from `./test_cases.yml`.

### Test config

Each test configuration yaml should contain the following attributes:

| Attributes          | Description                                                 |
|---------------------|-------------------------------------------------------------|
| `system_setup`      | How to set up the system. (num of servers, clients)         |
| `single_app_as_job` | Whether the test cases are single app job or not.           |
| `cleanup`           | Whether to clean up test folders or not. (Default to True.) |
| `tests`             | The test cases to run                                       |

If `single_app_as_job` is True, `apps_root_dir` is required.
Otherwise, `jobs_root_dir` is required.

### System setup config

Each system setup configuration yaml should contain the following attribute:

| Attributes     | Description                                                          |
|----------------|----------------------------------------------------------------------|
| poc            | Which poc folder to use (to get fed_server.json and fed_client.json) |
| n_servers      | number of servers                                                    |
| n_clients      | number of client sites                                               |
| snapshot_path  | Where to store server snapshot. (needs to match what's inside poc)   |
| job_store_path | Where to store job information. (needs to match what's inside poc)   | 


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

- data:
  - apps: applications for testing
  - ha: ha test event sequence yaml files
  - single_app_as_job: test configuration for single app as job test cases
  - system: system setup config yaml files
- tf2: TensorFlow2 related codes for the applications used in
  integration tests.
- validators: Codes that implement the logic to validate the running result
  once the job is finished.

### apps

Because the applications inside the `apps` folder is treated as single app job.

Each application in apps folder should contain `config` folder.

And should have `config_fed_server.json` and `config_fed_client.json` inside the config folder.
