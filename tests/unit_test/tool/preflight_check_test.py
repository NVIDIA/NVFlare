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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.tool.preflight_check import check_packages


def test_check_packages_does_not_double_exit_after_output_ok(tmp_path):
    (tmp_path / "startup").mkdir()
    args = MagicMock(package_path=str(tmp_path))

    checker = MagicMock()
    checker.should_be_checked.return_value = False
    checker.check.return_value = 0

    with patch("nvflare.tool.cli_schema.handle_schema_flag"):
        with patch("nvflare.tool.preflight_check.ServerPackageChecker", return_value=checker):
            with patch("nvflare.tool.preflight_check.ClientPackageChecker", return_value=checker):
                with patch("nvflare.tool.preflight_check.NVFlareConsolePackageChecker", return_value=checker):
                    with patch("nvflare.tool.cli_output.output_ok", side_effect=SystemExit(1)) as mock_output_ok:
                        with pytest.raises(SystemExit) as exc_info:
                            check_packages(args)

    assert exc_info.value.code == 1
    mock_output_ok.assert_called_once()
