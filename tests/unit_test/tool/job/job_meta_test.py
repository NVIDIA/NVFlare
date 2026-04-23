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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound
from nvflare.tool import cli_output


class TestJobMeta:
    """Tests for nvflare job meta command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, job_id="abc123", output="json"):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        return args

    def test_meta_success_json(self, capsys):
        """job meta success: returns job metadata in JSON envelope."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args()
        meta = {"job_id": "abc123", "name": "my_job", "status": "FINISHED_OK"}
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = meta

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_meta(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"

    def test_meta_not_found_exits_1(self):
        """job meta JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args(job_id="notfound")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_meta(args)
        assert exc_info.value.code == 1

    def test_meta_returns_none_exits_1(self):
        """When get_job_meta returns None, exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args(job_id="missing")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = None

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_meta(args)
        assert exc_info.value.code == 1

    def test_meta_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_meta

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_meta(args)

    def test_meta_parser_positional_job_id(self):
        """meta parser should accept positional job_id."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["meta"]
        assert parser is not None
        args = parser.parse_args(["abc123"])
        assert args.job_id == "abc123"

    def test_meta_parser_accepts_startup_target(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["meta"]
        args = parser.parse_args(["abc123", "--startup-target", "prod"])
        assert args.startup_target == "prod"
