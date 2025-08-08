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

import json
import os
import shlex
import subprocess
import sys
import time
from typing import List, Optional

import pytest

from tests.integration_test.src import ProvisionSiteLauncher
from tests.integration_test.src.constants import PREFLIGHT_CHECK_SCRIPT


class JSONPreflightResult:
    """Wrapper for JSON preflight check results."""

    def __init__(self, data: dict):
        self.data = data
        # Get the first (and usually only) package result
        self.package_path = list(data.keys())[0] if data else None
        self.package_data = data.get(self.package_path, {}) if self.package_path else {}
        self.checks = self.package_data.get("checks", [])

    def get_check(self, name_part: str) -> Optional[dict]:
        """Get a check that contains the given name part."""
        for check in self.checks:
            if name_part.lower() in check["name"].lower():
                return check
        return None

    def assert_check_passed(self, name_part: str, message: str = None):
        """Assert a specific check passed with detailed error info."""
        check = self.get_check(name_part)
        if not check:
            available = [c["name"] for c in self.checks]
            pytest.fail(f"Check containing '{name_part}' not found. Available: {available}")

        if not check["passed"]:
            error_msg = message or f"Check '{check['name']}' failed"
            error_msg += f"\n  Status: {check['status']}"
            error_msg += f"\n  Solution: {check['solution']}"
            pytest.fail(error_msg)

    def assert_all_passed(self, message: str = None):
        """Assert all checks passed."""
        if not self.package_data.get("all_passed", False):
            failed = [c for c in self.checks if not c["passed"]]
            error_msg = message or f"Some checks failed ({len(failed)}/{len(self.checks)})"
            for check in failed:
                error_msg += f"\n  - {check['name']}: {check['status']}"
            pytest.fail(error_msg)

    def get_failed_checks(self) -> List[dict]:
        """Get list of failed checks."""
        return [c for c in self.checks if not c["passed"]]


TEST_CASES = [
    {"project_yaml": "data/projects/dummy.yml", "admin_name": "super@test.org"},
]

SERVER_START_TIME = 15


def _run_preflight_check_json(package_path: str) -> JSONPreflightResult:
    """Run preflight check with JSON output."""
    command = f"{sys.executable} -m {PREFLIGHT_CHECK_SCRIPT} -p {package_path} --json"
    print(f"Executing command: {command}")

    try:
        output = subprocess.check_output(shlex.split(command), stderr=subprocess.PIPE, text=True)

        # Extract JSON from mixed output
        lines = output.strip().split("\n")

        # Try to find complete JSON objects
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and non-JSON lines
            if not line or any(msg in line for msg in ["killing", "killed", "process"]):
                i += 1
                continue

            # Look for JSON object start
            if line.startswith("{"):
                # Found start of JSON, now find the complete object
                json_lines = []
                brace_count = 0
                j = i

                while j < len(lines):
                    current_line = lines[j]
                    json_lines.append(current_line)

                    # Count braces to find complete object
                    brace_count += current_line.count("{") - current_line.count("}")

                    if brace_count == 0:
                        # Complete JSON object
                        json_str = "\n".join(json_lines)
                        try:
                            # Skip empty objects
                            if json_str.strip() != "{}":
                                data = json.loads(json_str)
                                if data:  # Only return non-empty data
                                    return JSONPreflightResult(data)
                        except json.JSONDecodeError:
                            pass
                        break
                    j += 1
            i += 1

        # If we get here, no valid JSON was found
        pytest.fail(f"No valid JSON found in output. Raw output:\n{output}")

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Preflight check command failed: {e.stderr}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing output: {e}\nOutput: {output}")


@pytest.fixture(params=TEST_CASES)
def setup_system(request):
    test_config = request.param
    project_yaml_path = test_config["project_yaml"]
    admin_name = test_config["admin_name"]

    if not os.path.isfile(project_yaml_path):
        raise RuntimeError(f"Missing project_yaml at {project_yaml_path}.")

    site_launcher = ProvisionSiteLauncher(project_yaml=project_yaml_path)
    workspace_root = site_launcher.prepare_workspace()
    print(f"Workspace root is {workspace_root}")

    admin_folder_root = os.path.abspath(os.path.join(workspace_root, admin_name))
    return site_launcher, admin_folder_root


@pytest.mark.xdist_group(name="preflight_tests_group")
class TestPreflightCheckJSON:
    """Preflight check tests using JSON output for clean validation."""

    def test_server_preflight_with_overseer(self, setup_system):
        """Test server preflight checks when overseer is running."""
        site_launcher, _ = setup_system
        try:
            for server_name, server_props in site_launcher.server_properties.items():
                result = _run_preflight_check_json(server_props.root_dir)

                # Verify all expected server checks
                result.assert_check_passed("overseer running")
                result.assert_check_passed("grpc port binding")
                result.assert_check_passed("admin port binding")
                result.assert_check_passed("snapshot storage writable")
                result.assert_check_passed("job storage writable")
                result.assert_check_passed("dry run")

                result.assert_all_passed(f"All {server_name} checks should pass")

        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_server_preflight_without_overseer(self, setup_system):
        """Test server preflight checks when overseer is not running."""
        site_launcher, _ = setup_system
        try:
            for server_name, server_props in site_launcher.server_properties.items():
                result = _run_preflight_check_json(server_props.root_dir)
                result.assert_all_passed(f"All {server_name} checks should pass with dummy overseer")
        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_client_preflight_check(self, setup_system):
        """Test client preflight checks."""
        site_launcher, _ = setup_system
        try:
            site_launcher.start_servers()
            time.sleep(SERVER_START_TIME)

            for client_name, client_props in site_launcher.client_properties.items():
                result = _run_preflight_check_json(client_props.root_dir)

                # Verify critical client checks
                result.assert_check_passed("overseer running")
                result.assert_check_passed("service provider list available")
                result.assert_check_passed("primary SP's socket server available")
                result.assert_check_passed("primary SP's GRPC server available")
                result.assert_check_passed("dry run")

                # Non-primary SP checks are optional - just warn if they fail
                failed_checks = result.get_failed_checks()
                optional_failures = [c for c in failed_checks if "non-primary SP" in c["name"]]
                if optional_failures:
                    for check in optional_failures:
                        print(f"Warning: Optional check failed for {client_name}: {check['name']} - {check['status']}")

                # Ensure no critical checks failed
                critical_failures = [c for c in failed_checks if "non-primary SP" not in c["name"]]
                assert (
                    not critical_failures
                ), f"Critical checks failed for {client_name}: {[(c['name'], c['status']) for c in critical_failures]}"

        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()

    def test_admin_console_preflight_check(self, setup_system):
        """Test admin console preflight checks."""
        site_launcher, admin_folder_root = setup_system
        try:
            site_launcher.start_servers()
            time.sleep(SERVER_START_TIME)

            result = _run_preflight_check_json(admin_folder_root)

            # Admin console has same checks as client
            result.assert_check_passed("overseer running")
            result.assert_check_passed("service provider list available")
            result.assert_check_passed("primary SP's socket server available")
            result.assert_check_passed("primary SP's GRPC server available")
            result.assert_check_passed("dry run")

            result.assert_all_passed("All admin console checks should pass")

        finally:
            site_launcher.stop_all_sites()
            site_launcher.cleanup()
