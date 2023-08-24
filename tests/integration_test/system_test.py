# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import sys
import tempfile
import time

import pytest

from tests.integration_test.src import (
    NVFTestDriver,
    NVFTestError,
    POCSiteLauncher,
    ProvisionSiteLauncher,
    cleanup_path,
    read_yaml,
    run_command_in_subprocess,
)


def get_module_class_from_full_path(full_path):
    tokens = full_path.split(".")
    cls_name = tokens[-1]
    mod_name = ".".join(tokens[: len(tokens) - 1])
    return mod_name, cls_name


def get_test_config(test_config_yaml: str):
    print(f"Test config from:  {test_config_yaml}")
    test_config = read_yaml(test_config_yaml)
    test_config["single_app_as_job"] = test_config.get("single_app_as_job", False)
    test_config["cleanup"] = test_config.get("cleanup", True)
    test_config["ha"] = test_config.get("ha", False)
    for x in ["cleanup", "single_app_as_job"]:
        if x not in test_config:
            raise NVFTestError(f"Test config: {test_config_yaml} missing required attributes {x}.")
        print(f"\t{x}: {test_config[x]}")

    if test_config["single_app_as_job"]:
        if "apps_root_dir" not in test_config:
            raise NVFTestError(f"Test config: {test_config_yaml} missing apps_root_dir.")
        print(f"\tapps_root_dir: {test_config['apps_root_dir']}")
    else:
        if "jobs_root_dir" not in test_config:
            raise NVFTestError(f"Test config: {test_config_yaml} missing jobs_root_dir.")
        print(f"\tjobs_root_dir: {test_config['jobs_root_dir']}")

    if test_config["ha"]:
        if "project_yaml" not in test_config:
            raise NVFTestError(f"Test config: {test_config_yaml} missing project_yaml.")
    else:
        for x in ["n_servers", "n_clients"]:
            if x not in test_config:
                raise NVFTestError(f"Test config: {test_config_yaml} missing required attributes {x}.")

    return test_config


framework = os.environ.get("NVFLARE_TEST_FRAMEWORK")
test_configs_yaml = "auto_test_configs.yml" if framework == "auto" else "test_configs.yml"
test_configs = read_yaml(test_configs_yaml)
if framework not in test_configs["test_configs"]:
    print(f"Framework/test {framework} is not supported, using default numpy.")
    framework = "numpy"
print(f"Testing framework {framework}")
test_configs = test_configs["test_configs"][framework]


@pytest.fixture(
    scope="class",
    params=test_configs,
)
def setup_and_teardown_system(request):
    yaml_path = os.path.join(os.path.dirname(__file__), request.param)
    test_config = get_test_config(yaml_path)

    cleanup = test_config["cleanup"]
    ha = test_config["ha"]
    poll_period = test_config.get("poll_period", 5)
    additional_python_paths = test_config.get("additional_python_paths", [])
    for additional_python_path in additional_python_paths:
        sys.path.append(os.path.abspath(additional_python_path))

    test_temp_dir = tempfile.mkdtemp()

    test_driver = None
    site_launcher = None
    try:
        if ha:
            project_yaml_path = test_config.get("project_yaml")
            if not os.path.isfile(project_yaml_path):
                raise NVFTestError(f"Missing project_yaml at {project_yaml_path}.")
            site_launcher = ProvisionSiteLauncher(project_yaml=project_yaml_path)
            poc = False
            super_user_name = "super@test.org"
        else:
            n_servers = int(test_config["n_servers"])
            if n_servers != 1:
                raise NVFTestError("POC mode can only use one server. For more servers, use HA with provisioned mode.")
            n_clients = int(test_config["n_clients"])
            site_launcher = POCSiteLauncher(n_servers=n_servers, n_clients=n_clients)
            poc = False  # POC now uses SSL as well so this needs to be False
            super_user_name = "admin@nvidia.com"

        workspace_root = site_launcher.prepare_workspace()
        print(f"Workspace root is {workspace_root}")
        print(f"sys.path start is: {sys.path}")

        if ha:
            site_launcher.start_overseer()
        site_launcher.start_servers()
        site_launcher.start_clients()

        # testing cases
        test_cases = []
        jobs_root_dir = test_config["jobs_root_dir"]
        for x in test_config["tests"]:
            test_cases.append(
                (
                    x["test_name"],
                    x.get("validators"),
                    x.get("setup", []),
                    x.get("teardown", []),
                    x.get("event_sequence", []),
                    x.get("reset_job_info", True),
                ),
            )

        download_root_dir = os.path.join(test_temp_dir, "download_result")
        os.mkdir(download_root_dir)
        test_driver = NVFTestDriver(
            site_launcher=site_launcher, download_root_dir=download_root_dir, poll_period=poll_period
        )
        test_driver.initialize_super_user(
            workspace_root_dir=workspace_root, upload_root_dir=jobs_root_dir, poc=poc, super_user_name=super_user_name
        )
        if ha:
            test_driver.initialize_admin_users(
                workspace_root_dir=workspace_root,
                upload_root_dir=jobs_root_dir,
                poc=poc,
                admin_user_names=site_launcher.admin_user_names,
            )
        test_driver.ensure_clients_started(num_clients=len(site_launcher.client_properties.keys()), timeout=2000)
        yield ha, test_cases, site_launcher, test_driver
    finally:
        if test_driver:
            test_driver.finalize()
        if site_launcher:
            site_launcher.stop_all_sites()
        if cleanup:
            if site_launcher:
                site_launcher.cleanup()
            cleanup_path(test_temp_dir)
        sys.path = sys.path[: -len(additional_python_paths) or None]
        print(f"sys.path finish is: {sys.path}")


@pytest.mark.xdist_group(name="system_tests_group")
class TestSystem:
    def test_run_job_complete(self, setup_and_teardown_system):
        ha, test_cases, site_launcher, test_driver = setup_and_teardown_system

        print(f"Server status: {test_driver.server_status()}.")
        print(f"Client status: {test_driver.client_status()}")

        test_validate_results = []
        for test_data in test_cases:
            test_name, validators, setup, teardown, event_sequence, reset_job_info = test_data
            print(f"Running test {test_name}")

            start_time = time.time()
            for command in setup:
                print(f"Running setup command: {command}")
                process = run_command_in_subprocess(command)
                process.wait()

            test_driver.run_event_sequence(event_sequence)

            job_result = None
            if test_driver.job_id is not None:
                job_result = test_driver.get_job_result(test_driver.job_id)

            # Get the job validator
            if validators:
                validate_result = True
                for validator in validators:
                    validator_module = validator["path"]
                    validator_args = validator.get("args", {})
                    # Create validator instance
                    module_name, class_name = get_module_class_from_full_path(validator_module)
                    job_validator_cls = getattr(importlib.import_module(module_name), class_name)
                    job_validator = job_validator_cls(**validator_args)

                    job_validate_res = job_validator.validate_results(
                        job_result=job_result,
                        client_props=list(site_launcher.client_properties.values()),
                    )
                    print(f"Test {test_name}, Validator {job_validator.__class__.__name__}, Result: {job_validate_res}")
                    if not job_validate_res:
                        validate_result = False
                        break
            else:
                print("No validators provided so results set to No Validators.")
                validate_result = "No Validators"
            test_validate_results.append((test_name, validate_result))

            print(f"Finished running test '{test_name}' in {time.time() - start_time} seconds.")
            for command in teardown:
                print(f"Running teardown command: {command}")
                process = run_command_in_subprocess(command)
                process.wait()
            test_driver.reset_test_info(reset_job_info=reset_job_info)
            print("\n\n\n\n\n")

        _print_validate_result(validate_result=test_validate_results)


def _print_validate_result(validate_result: list):
    test_name_length = 10
    result_length = 20
    failure = False
    for test_name, result in validate_result:
        test_name_length = max(test_name_length, len(test_name))
        result_length = max(result_length, len(str(result)))
        if not result:
            failure = True
    print("=" * (test_name_length + result_length + 7))
    print("| {arg:<{width}s} |".format(arg="Test validate results", width=test_name_length + result_length + 3))
    print("|" + "-" * (test_name_length + result_length + 5) + "|")
    print(
        "| {test_name:<{width1}s} | {result:<{width2}s} |".format(
            test_name="Test Name",
            result="Validate Result",
            width1=test_name_length,
            width2=result_length,
        )
    )
    print("|" + "-" * (test_name_length + result_length + 5) + "|")
    for test_name, result in validate_result:
        print(
            "| {test_name:<{width1}s} | {result:<{width2}s} |".format(
                test_name=test_name,
                result=str(result),
                width1=test_name_length,
                width2=result_length,
            )
        )
    print("|" + "-" * (test_name_length + result_length + 5) + "|")
    print("| {arg:<{width}s} |".format(arg=f"Final result: {not failure}", width=test_name_length + result_length + 3))
    print("=" * (test_name_length + result_length + 7))
    assert not failure
