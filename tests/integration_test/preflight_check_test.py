# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pty
import shlex
import subprocess
import sys
import time
from io import BytesIO

import pytest

from tests.integration_test.src import ProvisionSiteLauncher
from tests.integration_test.src.constants import PREFLIGHT_CHECK_SCRIPT

TEST_CASES = [
    {"project_yaml": "data/projects/dummy.yml", "admin_name": "super@test.org", "is_dummy_overseer": True},
]

SERVER_OUTPUT_PASSED = [
    "-----------------------------------------------------------------------------------------------------------------------------------",
    "| Checks                          | Problems                                                                         | How to fix |",
    "|---------------------------------------------------------------------------------------------------------------------------------|",
    "| Check grpc port binding         | PASSED                                                                           | N/A        |",
    "|---------------------------------------------------------------------------------------------------------------------------------|",
    "| Check admin port binding        | PASSED                                                                           | N/A        |",
    "|---------------------------------------------------------------------------------------------------------------------------------|",
    "| Check snapshot storage writable | PASSED                                                                           | N/A        |",
    "|---------------------------------------------------------------------------------------------------------------------------------|",
    "| Check job storage writable      | PASSED                                                                           | N/A        |",
    "|---------------------------------------------------------------------------------------------------------------------------------|",
    "| Check dry run                   | PASSED                                                                           | N/A        |",
    "-----------------------------------------------------------------------------------------------------------------------------------",
]

CLIENT_OUTPUT_PASSED = [
    "-------------------------------------------------------------------------------------------------------------------------------",
    "| Checks                      | Problems                                                                         | How to fix |",
    "|-----------------------------------------------------------------------------------------------------------------------------|",
    "| Check GRPC server available | PASSED                                                                           | N/A        |",
    "|-----------------------------------------------------------------------------------------------------------------------------|",
    "| Check dry run               | PASSED                                                                           | N/A        |",
    "-------------------------------------------------------------------------------------------------------------------------------",
]

SERVER_START_TIME = 15


def _filter_output(output):
    lines = []
    for line in output.decode("utf-8").splitlines():
        if "Checking Package" in line:
            continue
        elif "killing dry run process" in line:
            continue
        elif "killed dry run process" in line:
            continue
        elif not line:
            continue
        lines.append(line)
    return lines


def _run_preflight_check_command_in_subprocess(package_path: str):
    command = f"{sys.executable} -m {PREFLIGHT_CHECK_SCRIPT} -p {package_path}"
    print(f"Executing command {command} in subprocess")
    output = subprocess.check_output(shlex.split(command))
    return output


def _run_preflight_check_command_in_pseudo_terminal(package_path: str):
    command = f"{sys.executable} -m {PREFLIGHT_CHECK_SCRIPT} -p {package_path}"
    print(f"Executing command {command} in pty")

    with BytesIO() as output:

        def read(fd):
            data = os.read(fd, 1024 * 1024 * 1024)
            output.write(data)
            return data

        pty.spawn(shlex.split(command), read)

        return output.getvalue()


def _run_preflight_check_command(package_path: str, method: str = "subprocess"):
    if method == "subprocess":
        return _run_preflight_check_command_in_subprocess(package_path)
    else:
        return _run_preflight_check_command_in_pseudo_terminal(package_path)


@pytest.fixture(
    params=TEST_CASES,
)
def setup_system(request):
    test_config = request.param
    project_yaml_path = test_config["project_yaml"]
    is_dummy_overseer = test_config["is_dummy_overseer"]
    admin_name = test_config["admin_name"]

    if not os.path.isfile(project_yaml_path):
        raise RuntimeError(f"Missing project_yaml at {project_yaml_path}.")
    site_launcher = ProvisionSiteLauncher(project_yaml=project_yaml_path)
    workspace_root = site_launcher.prepare_workspace()
    print(f"Workspace root is {workspace_root}")

    admin_folder_root = os.path.abspath(os.path.join(workspace_root, admin_name))

    return site_launcher, is_dummy_overseer, admin_folder_root


@pytest.mark.xdist_group(name="preflight_tests_group")
class TestPreflightCheck:
    def test_run_check_on_server_after_overseer_start(self, setup_system):
        site_launcher, is_dummy_overseer, _ = setup_system
        try:
            if not is_dummy_overseer:
                site_launcher.start_overseer()
            # preflight-check on server
            for server_name, server_props in site_launcher.server_properties.items():
                output = _run_preflight_check_command(package_path=server_props.root_dir)
                filtered_output = _filter_output(output)
                assert filtered_output == SERVER_OUTPUT_PASSED
        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_run_check_on_server_before_overseer_start(self, setup_system):
        site_launcher, is_dummy_overseer, _ = setup_system
        try:
            # preflight-check on server
            for server_name, server_props in site_launcher.server_properties.items():
                output = _run_preflight_check_command(package_path=server_props.root_dir)
                filtered_output = _filter_output(output)
                if is_dummy_overseer:
                    assert filtered_output == SERVER_OUTPUT_PASSED
                else:
                    assert filtered_output != SERVER_OUTPUT_PASSED
        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_run_check_on_client(self, setup_system):
        site_launcher, is_dummy_overseer, _ = setup_system
        filtered_output = None
        try:
            if not is_dummy_overseer:
                site_launcher.start_overseer()
            site_launcher.start_servers()
            time.sleep(SERVER_START_TIME)

            # preflight-check on clients
            for client_name, client_props in site_launcher.client_properties.items():
                output = _run_preflight_check_command(package_path=client_props.root_dir)
                filtered_output = _filter_output(output)
                assert filtered_output == CLIENT_OUTPUT_PASSED
        except Exception:
            raise
        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_run_check_on_admin_console(self, setup_system):
        site_launcher, is_dummy_overseer, admin_folder_root = setup_system
        filtered_output = None
        try:
            if not is_dummy_overseer:
                site_launcher.start_overseer()
            site_launcher.start_servers()
            time.sleep(SERVER_START_TIME)

            # preflight-check on admin console
            output = _run_preflight_check_command(package_path=admin_folder_root)
            filtered_output = _filter_output(output)
            assert filtered_output == CLIENT_OUTPUT_PASSED
        except Exception:
            raise
        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()
