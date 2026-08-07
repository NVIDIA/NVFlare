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

import shlex
import subprocess
import sys

import pytest

from tests.integration_test.src import utils


def test_run_command_and_check_success_streams_output(capfd):
    command = shlex.join([sys.executable, "-c", "print('command-ok')"])

    utils.run_command_and_check(command)

    captured = capfd.readouterr()
    assert "command-ok" in captured.out


def test_run_command_and_check_failure_reports_command_exit_and_output(capfd):
    child_code = "import sys; print('stdout-sentinel'); print('stderr-sentinel', file=sys.stderr); sys.exit(23)"
    command = shlex.join([sys.executable, "-c", child_code])

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        utils.run_command_and_check(command)

    captured = capfd.readouterr()
    assert exc_info.value.returncode == 23
    assert exc_info.value.cmd == shlex.split(command)
    assert "exit status 23" in str(exc_info.value)
    assert "stdout-sentinel" in captured.out
    assert "stderr-sentinel" in captured.err
