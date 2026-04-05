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


class TestJobDownload:
    """Tests for nvflare job download command."""

    def _make_args(self, job_id="abc123", output="json", output_dir="./"):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        args.output_dir = output_dir
        return args

    def test_download_success_json(self, capsys):
        """job download success: JSON envelope has job_id and path."""
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.download_job_result.return_value = "/path/to/results"

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_download(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["job_id"] == "abc123"
        assert "path" in data["data"]

    def test_download_with_output_dir(self):
        """output_dir is used as destination."""
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args(output_dir="/my/results")
        mock_sess = MagicMock()
        mock_sess.download_job_result.return_value = "/path/to/results"

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_download(args)

        import os

        expected_dest = os.path.abspath("/my/results")
        mock_sess.download_job_result.assert_called_once_with("abc123", expected_dest)

    def test_download_not_found_exits_1(self):
        """JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args(job_id="notfound")
        mock_sess = MagicMock()
        mock_sess.download_job_result.side_effect = Exception("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_download(args)
        assert exc_info.value.code == 1

    def test_download_parser(self):
        """download parser should accept job_id and -o flag."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["download"]
        assert parser is not None
        args = parser.parse_args(["abc123", "-o", "/tmp/results"])
        assert args.job_id == "abc123"
        assert args.output_dir == "/tmp/results"
