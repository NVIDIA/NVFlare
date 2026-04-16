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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection
from nvflare.tool import cli_output


def _make_args(job_id="abc123", site="all"):
    args = MagicMock()
    args.job_id = job_id
    args.site = site
    return args


class TestJobStats:
    """Tests for nvflare job stats command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _fake_session(self, mock_sess):
        @contextmanager
        def _ctx():
            yield mock_sess

        return _ctx

    def test_stats_json_envelope_shape(self, capsys):
        """show_stats result is wrapped in job_id + stats envelope."""
        from nvflare.tool.job.job_cli import cmd_job_stats

        stats_payload = {"server": {"round": 5}}
        mock_sess = MagicMock()
        mock_sess.show_stats.return_value = stats_payload

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_stats(_make_args(job_id="abc123"))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        data = envelope["data"]
        assert data["job_id"] == "abc123"
        assert data["stats"] == stats_payload

    def test_stats_job_not_found_exits_1(self, capsys):
        """JobNotFound maps to JOB_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_stats

        mock_sess = MagicMock()
        mock_sess.show_stats.side_effect = JobNotFound("job does not exist")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_stats(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "JOB_NOT_FOUND"
        assert envelope["exit_code"] == 1

    def test_stats_job_not_found_via_not_found_phrase(self, capsys):
        """JobNotFound also maps to JOB_NOT_FOUND."""
        from nvflare.tool.job.job_cli import cmd_job_stats

        mock_sess = MagicMock()
        mock_sess.show_stats.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_stats(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "JOB_NOT_FOUND"

    def test_stats_connection_failed_exits_2(self, capsys):
        """NoConnection maps to CONNECTION_FAILED, exit 2."""
        from nvflare.tool.job.job_cli import cmd_job_stats

        mock_sess = MagicMock()
        mock_sess.show_stats.side_effect = NoConnection("connection refused")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_stats(_make_args())
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert envelope["exit_code"] == 2

    def test_stats_parser(self):
        """'stats' subparser parses positional job_id correctly."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["stats"]
        assert parser is not None
        args = parser.parse_args(["abc123"])
        assert args.job_id == "abc123"
