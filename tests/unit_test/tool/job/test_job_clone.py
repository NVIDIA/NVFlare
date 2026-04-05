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
from unittest.mock import MagicMock, patch

import pytest


class TestJobClone:
    """Tests for nvflare job clone command."""

    def _make_args(self, job_id="abc123", output="json"):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        return args

    def test_clone_success_json(self, capsys):
        """job clone success: JSON envelope has source_job_id and new_job_id."""
        from nvflare.tool.job.job_cli import cmd_job_clone

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.clone_job.return_value = "def456"

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_clone(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["source_job_id"] == "abc123"
        assert data["data"]["new_job_id"] == "def456"

    def test_clone_not_found_exits_1(self):
        """job clone with unknown job exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_clone

        args = self._make_args(job_id="notfound")
        mock_sess = MagicMock()
        mock_sess.clone_job.side_effect = Exception("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_clone(args)
        assert exc_info.value.code == 1

    def test_clone_parser_positional_job_id(self):
        """clone parser should accept positional job_id."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["clone"]
        assert parser is not None
        args = parser.parse_args(["abc123"])
        assert args.job_id == "abc123"
