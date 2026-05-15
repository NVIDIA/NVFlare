# NVIDIA FLARE Integration test

## Setup

Some integration test configs use local host aliases when provisioning the test system.
That requires `localhost0` to map to `127.0.0.1`.
You need to either modify the `/etc/hosts` file before running the test,
or, if you are running in a docker container, use `--add-host localhost0:127.0.0.1`.

From the repo root, install NVFlare and set the local test environment:

```bash
python -m pip install -e .[dev]
export PYTHONPATH=$PWD
export GRPC_POLL_STRATEGY=poll
export GRPC_ENABLE_FORK_SUPPORT=False
```

## Run

Direct pytest suites are grouped by expected CI cadence:

```bash
cd tests/integration_test
python -m pytest -v --log-cli-level=INFO --capture=no fast
python -m pytest -v --log-cli-level=INFO --capture=no slow
```

The slow suite includes XGBoost recipe tests that require the federated XGBoost wheel. CI installs it
through `ci/run_integration.sh slow`. For direct local pytest runs, install the wheel from
`examples/advanced/xgboost/requirements.txt` first, or run the suite through `ci/run_integration.sh slow`.

Run one direct pytest suite:

```bash
python -m pytest -v --log-cli-level=INFO --capture=no fast/study_session_test.py
python -m pytest -v --log-cli-level=INFO --capture=no slow/preflight_check_test.py
python -m pytest -v --log-cli-level=INFO --capture=no slow/experiment_tracking_recipes_test.py
python -m pytest -v --log-cli-level=INFO --capture=no slow/distributed_provisioning_test.py
python -m pytest -v --log-cli-level=INFO --capture=no slow/recipe_system_test.py
python -m pytest -v --log-cli-level=INFO --capture=no slow/xgb_histogram_recipe_test.py slow/xgb_vertical_recipe_test.py
```

Run config-driven system tests by selecting a `test_configs.yml` group:

```bash
NVFLARE_TEST_FRAMEWORK=numpy python -m pytest -v --log-cli-level=INFO --capture=no system_test.py
NVFLARE_TEST_FRAMEWORK=pytorch python -m pytest -v --log-cli-level=INFO --capture=no system_test.py
NVFLARE_TEST_FRAMEWORK=client_api python -m pytest -v --log-cli-level=INFO --capture=no system_test.py
```

Add `--junitxml=./integration_test.xml` to any command when you need a JUnit report.

Config-driven `system_test.py` event sequences time out after 1800 seconds by default.
Set `NVFLARE_EVENT_SEQUENCE_TIMEOUT` to override this globally, or set `event_sequence_timeout`
in a test config YAML. Use `0` to disable the harness timeout.

CI uses `ci/run_integration.sh` for environment setup and mode dispatch. This directory intentionally
does not provide a second local wrapper script.

## Test structure

The integration tests have these main entry paths:
  - The integration tests entry file is `tests/integration_test/system_test.py`.
    It reads the selected system-test config group from `./test_configs.yml`.
  - The premerge pytest entry points live under `tests/integration_test/fast/`.
  - The nightly pytest entry points live under `tests/integration_test/slow/`.
  - Helper scripts used by test setup commands live under `tests/integration_test/tools/`.

`test_configs.yml` is only for `system_test.py` configuration. Direct pytest suites should be
placed under `fast/` or `slow/`.

### Test configuration

Each test configuration YAML defines a whole FL system.
The `system_test.py` will read and parse the config to determine which `SiteLauncher` to use
to set up the whole system.

The test configuration YAML files fall into two groups: Proof-Of-Concept (POC)
configs and provisioned-system configs.

1. Required attributes for POC-type test config:

| Attributes        | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `n_servers`       | Number of servers                                           |
| `n_clients`       | Number of clients                                           |
| `cleanup`         | Whether to clean up test folders or not. (Default to True.) |
| `jobs_root_dir`   | The directory that contains the job folders to upload       |
| `tests`           | The test cases to run                                       |

An example would be `tests/integration_test/data/test_configs/one_job/test_hello_numpy.yml`.

2. Required attributes for provisioned-system test config:

| Attributes      | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| `project_yaml`  | The file that would be passed to NVFlare provision script to generate the startup kits |
| `poll_period`   | The polling period of `NVFTestDriver`. (Default to 5 seconds.)                          |
| `cleanup`       | Whether to clean up test folders or not. (Default to True.)                            |
| `jobs_root_dir` | The directory that contains the job folders to upload                                  |
| `tests`         | The test cases to run                                                                  |

Optional attributes for both config types:

| Attributes                 | Description                                                              |
|----------------------------|--------------------------------------------------------------------------|
| `event_sequence_timeout`   | Max seconds to wait for one event sequence. (Default to 1800 seconds.)   |

An example would be `tests/integration_test/data/test_configs/authorization/list_job.yml`.

### Test cases

Each test case has the following attributes:

| Attributes            | Description                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------|
| test_name             | Name of this test case.                                                                    |
| event_sequence        | What events to run for this test case.                                                     |
| validators (optional) | Which validator to use to validate the running result once the event sequence is finished. |
| setup (optional)      | What shell command to run before this test case.                                           |
| teardown (optional)   | What shell command to run after this test case.                                            |


The most important part is the "event_sequence", which is triggered one by one.

After all events in event_sequence is triggered, then this test case is done.

We will explain in details in the following section.

An example test case is shown:

```yaml
  - test_name: "run hello-pt"
    event_sequence:
      - "trigger":
          "type": "server_log"
          "data": "Server started"
        "actions": [ "submit_job hello-pt" ]
        "result":
          "type": "run_state"
          "data": { }
      - "trigger":
          "type": "run_state"
          "data": { "run_finished": True }
        "actions": [ "ensure_current_job_done" ]
        "result":
          "type": "run_state"
          "data": { "run_finished": True }
    validators:
      - path: tests.integration_test.src.validators.PTModelValidator
      - path: tests.integration_test.src.validators.CrossValResultValidator
        args: { server_model_names: [ "server" ] }
    setup:
      - python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='~/data', download=True)"
    teardown:
      - rm -rf ~/data

```

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

```text
tests/integration_test/
  README.md
  test_configs.yml
  fast/
    study_session_test.py
  slow/
    preflight_check_test.py
    experiment_tracking_recipes_test.py
    distributed_provisioning_test.py
    recipe_system_test.py
    xgb_histogram_recipe_test.py
    xgb_vertical_recipe_test.py
  system_test.py
  tools/
    export_recipe_job.py
    convert_to_test_job.py
    install_requirements.py
  src/
  data/
```

- test_configs.yml: `system_test.py` config selector only
- fast: premerge pytest suites
- slow: nightly pytest suites
- system_test.py: config-driven integration test runner
- tools: helper scripts used by integration test setup commands
- src: source codes for the integration test system:
  - action_handlers.py: define how to handle event actions.
  - constants.py: define constants shared by the test system.
  - nvf_test_driver.py: the test driver controls and coordinates the test system.
  - poc_site_launcher.py: site launcher implementation for Proof-Of-Concept mode.
  - provision_site_launcher.py: site launcher implementation that utilizes NVFlare provision.
  - site_launcher.py: base class of a site launcher.
  - utils.py: utility functions used by other files.
- data:
  - apps: applications for testing; Each folder in the apps folder should be a valid NVFlare application.
  - projects: project configurations for feed into NVFlare provision
  - test_configs: test configurations root folder:
    - authorization: test configurations for authorization
    - one_job: test configurations for simple 1 app job run
- validators: Codes that implement the logic to validate the running result
  once the job is finished
