# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import patch

from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckServerAvailable
from nvflare.tool.package_checker.nvflare_console_package_checker import NVFlareConsolePackageChecker
from nvflare.tool.package_checker.package_checker import CheckStatus, PackageChecker
from nvflare.tool.package_checker.utils import NVFlareRole


def _write_ephemeral_admin_config(package_dir):
    startup = package_dir / "startup"
    startup.mkdir()
    (startup / "fed_admin.json").write_text(
        json.dumps(
            {
                "admin": {
                    "host": "localhost",
                    "port": 8003,
                    "scheme": "grpc",
                    "ephemeral_admin_cert": {"provider": "step_ca"},
                }
            }
        ),
        encoding="utf-8",
    )


def test_server_check_uses_non_interactive_reachability_for_ephemeral_admin(tmp_path):
    _write_ephemeral_admin_config(tmp_path)
    rule = CheckServerAvailable(name="Check server available", role=NVFlareRole.ADMIN)

    with patch(
        "nvflare.tool.package_checker.check_rule.check_socket_server_running", return_value=True
    ) as socket_check:
        with patch("nvflare.tool.package_checker.check_rule.check_grpc_server_running") as grpc_check:
            result = rule(str(tmp_path), data=None)

    assert result.problem == CHECK_PASSED
    socket_check.assert_called_once_with(startup=str(tmp_path / "startup"), host="localhost", port=8003, scheme="tcp")
    grpc_check.assert_not_called()


def test_console_check_skips_interactive_dry_run_for_ephemeral_admin(tmp_path):
    _write_ephemeral_admin_config(tmp_path)
    checker = NVFlareConsolePackageChecker()
    checker.init(str(tmp_path))

    with patch.object(PackageChecker, "check_dry_run") as inherited_check:
        status = checker.check_dry_run()

    assert status == CheckStatus.PASS
    assert checker.report[str(tmp_path.resolve())][-1] == ("Check dry run", CHECK_PASSED, "N/A")
    inherited_check.assert_not_called()


def test_console_check_falls_back_to_normal_dry_run_for_invalid_config(tmp_path):
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "fed_admin.json").write_text("{", encoding="utf-8")
    checker = NVFlareConsolePackageChecker()
    checker.init(str(tmp_path))

    with patch.object(PackageChecker, "check_dry_run", return_value=CheckStatus.FAIL) as inherited_check:
        status = checker.check_dry_run()

    assert status == CheckStatus.FAIL
    inherited_check.assert_called_once_with()
