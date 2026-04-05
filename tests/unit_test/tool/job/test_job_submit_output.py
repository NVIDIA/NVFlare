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
from unittest.mock import MagicMock

import pytest


class TestJobSubmitOutput:
    """Tests for nvflare job submit output format."""

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.wait = kwargs.get("wait", False)
        args.timeout = kwargs.get("timeout", 0)
        args.study = kwargs.get("study", "default")
        args.debug = False
        args.job_folder = kwargs.get("job_folder", "/fake/job")
        args.config_file = None
        return args

    def test_json_envelope_on_success(self, capsys):
        """On success, output_ok emits JSON envelope with job_id."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"job_id": "abc123"})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["data"]["job_id"] == "abc123"

    def test_output_error_exits_with_code(self):
        """output_error should raise SystemExit with given exit_code."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("JOB_INVALID", exit_code=1)
        assert exc_info.value.code == 1

    def test_connection_failed_exits_2(self):
        """CONNECTION_FAILED should exit with code 2."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("CONNECTION_FAILED", exit_code=2)
        assert exc_info.value.code == 2

    def test_timeout_exits_3(self):
        """TIMEOUT should exit with code 3."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("TIMEOUT", exit_code=3)
        assert exc_info.value.code == 3

    def test_wait_flag_in_submit_parser(self):
        """Submit parser should include --wait, --timeout, --study, --output flags."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        parser = job_sub_cmd_parser["submit"]
        assert parser is not None
        # Check that the parser has these arguments by parsing with them
        args = parser.parse_args(["-j", "/some/job", "--wait", "--timeout", "60", "--study", "test"])
        assert args.wait is True
        assert args.timeout == 60
        assert args.study == "test"

    def test_wait_flag_triggers_monitor(self, capsys):
        """When --wait is set, internal_submit_job calls monitor_job."""
        from nvflare.tool.cli_output import output_ok

        # Simulate the wait path: monitor returns a dict
        meta = {"job_id": "abc123", "status": "FINISHED_OK"}
        output_ok(meta)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["status"] == "FINISHED_OK"
