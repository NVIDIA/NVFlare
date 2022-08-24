# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import shlex
import subprocess
import tempfile
import time

import pytest

from tests.integration_test.src import AdminController, POCSiteLauncher, ProvisionSiteLauncher
from tests.integration_test.utils import (
    cleanup_path,
    get_job_store_path_from_poc,
    get_snapshot_path_from_poc,
    read_yaml,
)

POC_PATH = "../../nvflare/poc"


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
    for x in ["n_servers", "n_clients", "cleanup", "single_app_as_job"]:
        if x not in test_config:
            raise RuntimeError(f"Test config: {test_config_yaml} missing required attributes {x}.")
        print(f"\t{x}: {test_config[x]}")

    if test_config["single_app_as_job"]:
        if "apps_root_dir" not in test_config:
            raise RuntimeError(f"Test config: {test_config_yaml} missing apps_root_dir.")
        print(f"\tapps_root_dir: {test_config['apps_root_dir']}")
    else:
        if "jobs_root_dir" not in test_config:
            raise RuntimeError(f"Test config: {test_config_yaml} missing jobs_root_dir.")
        print(f"\tjobs_root_dir: {test_config['jobs_root_dir']}")

    return test_config


def _cleanup(additional_paths=None, site_launcher=None):
    path_to_clean = additional_paths if additional_paths else []
    snapshot_path = get_snapshot_path_from_poc(POC_PATH)
    job_store_path = get_job_store_path_from_poc(POC_PATH)
    path_to_clean.append(snapshot_path)
    path_to_clean.append(job_store_path)
    if site_launcher:
        site_launcher.cleanup()

    for p in path_to_clean:
        cleanup_path(p)


test_configs = read_yaml("./test_cases.yml")
framework = os.environ.get("NVFLARE_TEST_FRAMEWORK", "numpy")
if framework not in ["numpy", "tensorflow", "pytorch"]:
    print(f"Framework {framework} is not supported, using default numpy.")
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

    cleaned = False
    cleanup_in_the_end = test_config["cleanup"]
    n_servers = int(test_config["n_servers"])
    n_clients = int(test_config["n_clients"])
    ha = test_config.get("ha", False)

    test_temp_dir = tempfile.mkdtemp()

    admin_controller = None
    site_launcher = None
    test_cases = []
    try:
        if not os.path.isdir(POC_PATH):
            raise RuntimeError(f"Missing POC folder at {POC_PATH}.")

        _cleanup()

        if ha:
            site_launcher = ProvisionSiteLauncher()
        else:
            site_launcher = POCSiteLauncher(poc_dir=POC_PATH)

        workspace_root = site_launcher.prepare_workspace(n_servers=n_servers, n_clients=n_clients)
        print(f"Workspace root is {workspace_root}")

        if ha:
            site_launcher.start_overseer()
        for i in range(n_servers):
            site_launcher.start_server(i)
            time.sleep(5)
        client_names = []
        for i in range(1, n_clients + 1):
            site_launcher.start_client(i)
            client_names.append(f"site-{i}")

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
                    x.get("event_sequence", ""),
                ),
            )

        download_root_dir = os.path.join(test_temp_dir, "download_result")
        os.mkdir(download_root_dir)
        admin_controller = AdminController(download_root_dir=download_root_dir)
        if not admin_controller.initialize(workspace_root_dir=workspace_root, upload_root_dir=jobs_root_dir, ha=ha):
            raise RuntimeError("AdminController init failed.")
        admin_controller.ensure_clients_started(num_clients=test_config["n_clients"])
    except RuntimeError:
        if admin_controller:
            admin_controller.finalize()
        if site_launcher:
            site_launcher.stop_all_sites()
        _cleanup([test_temp_dir], site_launcher)
        cleaned = True
    yield ha, test_cases, site_launcher, admin_controller
    admin_controller.finalize()
    site_launcher.stop_all_sites()
    if cleanup_in_the_end and not cleaned:
        _cleanup([test_temp_dir], site_launcher)


@pytest.mark.xdist_group(name="system_tests_group")
class TestSystem:
    def test_run_job_complete(self, setup_and_teardown_system):
        ha, test_cases, site_launcher, admin_controller = setup_and_teardown_system

        print(f"Server status: {admin_controller.server_status()}.")
        print(f"Client status: {admin_controller.client_status()}")

        test_validate_results = []
        for test_data in test_cases:
            test_name, validators, setup, teardown, event_sequence = test_data
            print(f"Running test {test_name}")

            start_time = time.time()
            for command in setup:
                print(f"Running setup command: {command}")
                process = subprocess.Popen(shlex.split(command))
                process.wait()

            admin_controller.run_event_sequence(site_launcher, event_sequence)

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

                    job_result = admin_controller.get_job_result(admin_controller.job_id)
                    job_validate_res = job_validator.validate_results(
                        job_result=job_result,
                        client_props=list(site_launcher.client_properties.values()),
                    )
                    print(f"Test {test_name}, Validator {job_validator.__class__.__name__}, Result: {job_validate_res}")
                    validate_result = validate_result and job_validate_res

                test_validate_results.append((test_name, validate_result))
            else:
                print("No validators provided so results won't be checked.")

            print(f"Finished running {test_name} in {time.time() - start_time} seconds.")
            for command in teardown:
                print(f"Running teardown command: {command}")
                process = subprocess.Popen(shlex.split(command))
                process.wait()

        print("==============================================================")
        print(f"Job validate results: {test_validate_results}")
        failure = False
        for job_name, job_result in test_validate_results:
            print(f"Job name: {job_name}, Result: {job_result}")
            if not job_result:
                failure = True
        print(f"Final result: {not failure}")
        print("==============================================================")

        assert not failure
