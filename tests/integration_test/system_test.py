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
import shlex
import shutil
import subprocess
import tempfile
import time

import pytest
import yaml

from tests.integration_test.admin_controller import AdminController
from tests.integration_test.oa_laucher import OALauncher
from tests.integration_test.site_launcher import POCDirectory, SiteLauncher
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


def get_system_setup(system_setup_yaml: str):
    """Gets system setup config from yaml."""
    system_setup = read_yaml(system_setup_yaml)
    print(f"System setup from {system_setup_yaml}:")
    system_setup["ha"] = system_setup.get("ha", False)
    for x in ["n_clients", "n_servers", "poc", "snapshot_path", "job_store_path", "ha"]:
        if x not in system_setup:
            raise RuntimeError(f"System setup: {system_setup_yaml} missing required attributes {x}.")
        print(f"\t{x}: {system_setup[x]}")
    return system_setup


def get_test_config(test_config_yaml: str):
    print(f"Test config from:  {test_config_yaml}")
    test_config = read_yaml(test_config_yaml)
    test_config["single_app_as_job"] = test_config.get("single_app_as_job", False)
    test_config["cleanup"] = test_config.get("cleanup", True)
    for x in ["system_setup", "cleanup", "single_app_as_job"]:
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


test_configs = read_yaml("./test_cases.yml")


@pytest.fixture(
    scope="class",
    params=test_configs["test_configs"],
)
def setup_and_teardown(request):
    yaml_path = os.path.join(os.path.dirname(__file__), request.param)
    test_config = get_test_config(yaml_path)
    system_setup = get_system_setup(test_config["system_setup"])

    cleanup = test_config["cleanup"]

    poc = system_setup["poc"]
    snapshot_path = system_setup["snapshot_path"]
    job_store_path = system_setup["job_store_path"]
    cleanup_path(snapshot_path)
    cleanup_path(job_store_path)

    ha = system_setup["ha"]
    poc = POCDirectory(poc_dir=poc, ha=ha)
    site_launcher = SiteLauncher(poc_directory=poc)
    if ha:
        site_launcher.start_overseer()
    site_launcher.start_servers(n=system_setup["n_servers"])
    site_launcher.start_clients(n=system_setup["n_clients"])

    test_temp_dir = tempfile.mkdtemp()
    # testing jobs
    test_jobs = []
    if test_config["single_app_as_job"]:
        jobs_root_dir = os.path.join(test_temp_dir, "generated_jobs")
        os.mkdir(jobs_root_dir)
        for x in test_config["tests"]:
            _ = generate_job_dir_for_single_app_job(
                app_name=x["app_name"],
                app_root_folder=test_config["apps_root_dir"],
                clients=[x.name for x in site_launcher.client_properties.values()],
                destination=jobs_root_dir,
                app_as_job=True,
            )
            test_jobs.append(
                (
                    x["app_name"],
                    x["validators"],
                    x.get("setup", []),
                    x.get("teardown", []),
                    x.get("event_sequence_yaml", ""),
                )
            )
    else:
        jobs_root_dir = test_config["jobs_root_dir"]
        for x in test_config["tests"]:
            test_jobs.append(
                (
                    x["job_name"],
                    x["validators"],
                    x.get("setup", []),
                    x.get("teardown", []),
                    x.get("event_sequence_yaml", ""),
                ),
            )

    download_root_dir = os.path.join(test_temp_dir, "download_result")
    os.mkdir(download_root_dir)
    admin_controller = AdminController(upload_root_dir=jobs_root_dir, download_root_dir=download_root_dir, ha=ha)
    if not admin_controller.initialize():
        raise RuntimeError("AdminController init failed.")
    admin_controller.ensure_clients_started(num_clients=system_setup["n_clients"])

    yield ha, test_jobs, site_launcher, admin_controller

    if admin_controller:
        admin_controller.finalize()
    if site_launcher:
        site_launcher.stop_all_sites()

        if cleanup:
            site_launcher.cleanup()

    if cleanup:
        cleanup_path(test_temp_dir)
        cleanup_path(snapshot_path)
        cleanup_path(job_store_path)


@pytest.mark.xdist_group(name="system_tests_group")
class TestSystem:
    @pytest.mark.skip(reason="skip due to no overseer in poc")
    def test_overseer_server_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(1)
            server_agent_list = oa_launcher.start_servers(2)
            client_agent_list = oa_launcher.start_clients(4)
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.pause_server(server_agent_list[0])
            time.sleep(15)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
            oa_launcher.resume_server(server_agent_list[0])
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server01"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()

    @pytest.mark.skip(reason="skip due to no overseer in poc")
    def test_overseer_client_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(10)
            server_agent_list = oa_launcher.start_servers(1)
            client_agent_list = oa_launcher.start_clients(1)
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.pause_client(client_agent_list[0])
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
            oa_launcher.resume_client(client_agent_list[0])
            time.sleep(10)
            psp = oa_launcher.get_primary_sp(client_agent_list[0])
            assert psp.name == "server00"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()

    @pytest.mark.skip(reason="skip due to no overseer in poc")
    def test_overseer_overseer_down_and_up(self):
        oa_launcher = OALauncher()
        try:
            oa_launcher.start_overseer()
            time.sleep(10)
            server_agent_list = oa_launcher.start_servers(1)
            client_agent_list = oa_launcher.start_clients(4)
            time.sleep(10)
            for client_agent in client_agent_list:
                psp = oa_launcher.get_primary_sp(client_agent)
                assert psp.name == "server00"
            oa_launcher.stop_overseer()
            time.sleep(10)
            oa_launcher.start_overseer()
            time.sleep(10)
            for client_agent in client_agent_list:
                psp = oa_launcher.get_primary_sp(client_agent)
                assert psp.name == "server00"
        finally:
            oa_launcher.stop_clients()
            oa_launcher.stop_servers()
            oa_launcher.stop_overseer()

    def test_run_job_complete(self, setup_and_teardown):
        ha, test_jobs, site_launcher, admin_controller = setup_and_teardown

        print(f"Server status: {admin_controller.server_status()}.")

        job_validate_results = []
        for job_data in test_jobs:
            start_time = time.time()

            test_job_name, validators, setup, teardown, event_sequence_yaml = job_data
            print(f"Running job {test_job_name}")
            for command in setup:
                print(f"Running setup command: {command}")
                process = subprocess.Popen(shlex.split(command))
                process.wait()

            admin_controller.submit_job(job_name=test_job_name)

            time.sleep(5)
            print(f"Server status after job submission: {admin_controller.server_status()}.")
            print(f"Client status after job submission: {admin_controller.client_status()}")

            if event_sequence_yaml:
                # admin_controller.run_app_ha(site_launcher, ha_tests["pt"][0])
                admin_controller.run_event_sequence(site_launcher, read_yaml(event_sequence_yaml))
            else:
                admin_controller.wait_for_job_done()

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

                    job_result = admin_controller.get_job_result()
                    job_validate_res = job_validator.validate_results(
                        job_result=job_result,
                        client_props=list(site_launcher.client_properties.values()),
                    )
                    print(
                        f"Job {test_job_name}, Validator {job_validator.__class__.__name__}, Result: {job_validate_res}"
                    )
                    validate_result = validate_result and job_validate_res

                job_validate_results.append((test_job_name, validate_result))
            else:
                print("No validators provided so results can't be checked.")

            print(f"Finished running {test_job_name} in {time.time() - start_time} seconds.")
            for command in teardown:
                print(f"Running teardown command: {command}")
                process = subprocess.Popen(shlex.split(command))
                process.wait()

        print("==============================================================")
        print(f"Job validate results: {job_validate_results}")
        failure = False
        for job_name, job_result in job_validate_results:
            print(f"Job name: {job_name}, Result: {job_result}")
            if not job_result:
                failure = True
        print(f"Final result: {not failure}")
        print("==============================================================")

        assert not failure
