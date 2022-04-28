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
import shutil
import sys
import time
import traceback

import pytest
import yaml

from tests.integration_test.admin_controller import AdminController
from tests.integration_test.site_launcher import POCDirectory, SiteLauncher
from tests.integration_test.test_ha import ha_tests
from tests.integration_test.utils import generate_job_dir_for_single_app_job


def get_module_class_from_full_path(full_path):
    tokens = full_path.split(".")
    cls_name = tokens[-1]
    mod_name = ".".join(tokens[: len(tokens) - 1])
    return mod_name, cls_name


def read_yaml(yaml_file_path):
    if not os.path.exists(yaml_file_path):
        raise RuntimeError(f"Yaml file doesnt' exist at {yaml_file_path}")

    with open(yaml_file_path, "rb") as f:
        data = yaml.safe_load(f)

    return data


def cleanup_path(path: str):
    if os.path.exists(path):
        print(f"Clean up directory: {path}")
        shutil.rmtree(path)


params = [
    # "./test_examples.yml",
    "./test_internal.yml",
    # "./test_ha.yml",
]


@pytest.fixture(
    scope="class",
    params=params,
)
def system_config(request):
    yaml_path = os.path.join(os.path.dirname(__file__), request.param)
    print("Loading params from ", yaml_path)
    data = read_yaml(yaml_path)
    for x in ["cleanup", "poc", "n_clients", "jobs_root_dir", "snapshot_path", "job_store_path"]:
        if x not in data:
            raise RuntimeError(f"YAML {yaml_path} missing required attributes {x}.")
    cleanup_path(data["snapshot_path"])
    cleanup_path(data["job_store_path"])
    return data


@pytest.mark.xdist_group(name="system_tests_group")
class TestSystem:
    def test_run_job_complete(self, system_config):
        site_launcher = None
        admin_controller = None

        cleanup = system_config["cleanup"]
        poc = system_config["poc"]
        n_clients = system_config["n_clients"]
        jobs_root_dir = system_config["jobs_root_dir"]
        snapshot_path = system_config["snapshot_path"]
        job_store_path = system_config["job_store_path"]
        ha = system_config.get("ha", False)
        try:
            print(f"cleanup = {cleanup}")
            print(f"poc = {poc}")
            print(f"n_clients = {n_clients}")
            print(f"jobs_root_dir = {jobs_root_dir}")
            print(f"snapshot_path = {snapshot_path}")
            print(f"job_store_path = {job_store_path}")

            poc = POCDirectory(poc_dir=poc)
            site_launcher = SiteLauncher(poc_directory=poc, ha=ha)
            if ha:
                site_launcher.start_overseer()
            site_launcher.start_servers(1)
            site_launcher.start_clients(n=n_clients)

            # testing jobs
            test_jobs = []
            generated_jobs = []
            for x in system_config["tests"]:
                if "job_name" in x:
                    test_jobs.append((x["job_name"], x["validators"]))
                    continue
                job = generate_job_dir_for_single_app_job(
                    app_name=x["app_name"],
                    app_root_folder=system_config["apps_root_dir"],
                    clients=[x["name"] for x in site_launcher.client_properties.values()],
                    destination=jobs_root_dir,
                )
                test_jobs.append((x["app_name"], x["validators"]))
                generated_jobs.append(job)

            admin_controller = AdminController(jobs_root_dir=jobs_root_dir, ha=ha)
            if not admin_controller.initialize():
                raise RuntimeError("AdminController init failed.")
            admin_controller.ensure_clients_started(num_clients=n_clients)

            print(f"Server status: {admin_controller.server_status()}.")

            job_results = []
            for job_data in test_jobs:
                start_time = time.time()

                test_job, validators = job_data

                print(f"Running job {test_job} with {validators}")

                admin_controller.submit_job(job_name=test_job)

                time.sleep(5)
                print(f"Server status after job submission: {admin_controller.server_status()}.")
                print(f"Client status after job submission: {admin_controller.client_status()}")

                if ha:
                    admin_controller.run_app_ha(site_launcher, ha_tests["pt"][0])
                else:
                    admin_controller.wait_for_job_done()

                server_data = site_launcher.get_server_data()
                client_data = site_launcher.get_client_data()
                run_data = admin_controller.get_run_data()

                # Get the app validator
                if validators:
                    validate_result = True
                    for validator_module in validators:
                        # Create validator instance
                        module_name, class_name = get_module_class_from_full_path(validator_module)
                        app_validator_cls = getattr(importlib.import_module(module_name), class_name)
                        app_validator = app_validator_cls()

                        print(server_data)
                        app_validate_res = app_validator.validate_results(
                            server_data=server_data,
                            client_data=client_data,
                            run_data=run_data,
                        )
                        print(
                            f"Job {test_job}, Validator {app_validator.__class__.__name__} result: {app_validate_res}"
                        )
                        validate_result = validate_result and app_validate_res

                    job_results.append((test_job, validate_result))
                else:
                    print("No validators provided so results can't be checked.")

                print(f"Finished running {test_job} in {time.time() - start_time} seconds.")

            print(f"Job results: {job_results}")
            failure = False
            for job_name, job_result in job_results:
                print(f"Job name: {job_name}, Result: {job_result}")
                if not job_result:
                    failure = True

            if cleanup:
                for job in generated_jobs:
                    shutil.rmtree(job)

            if failure:
                sys.exit(1)
        except BaseException as e:
            traceback.print_exc()
            print(f"Exception in test run: {e.__str__()}")
            raise ValueError("Tests failed") from e
        finally:
            if admin_controller:
                admin_controller.finalize()
            if site_launcher:
                site_launcher.stop_all_sites()

                if cleanup:
                    site_launcher.cleanup()
                    cleanup_path(snapshot_path)
                    cleanup_path(job_store_path)
