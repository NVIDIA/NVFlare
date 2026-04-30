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
import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
from nvflare.tool import cli_output


class TestJobDownload:
    """Tests for nvflare job download command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, job_id="abc123", output="json", output_dir=None):
        args = {"job_id": job_id, "output": output}
        if output_dir is not None:
            args["output_dir"] = output_dir
        return Namespace(**args)

    def _download_json(self, args, download_path, capsys):
        from nvflare.tool.job.job_cli import cmd_job_download

        mock_sess = MagicMock()
        mock_sess.download_job_result.return_value = str(download_path)

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_download(args)

        return mock_sess, json.loads(capsys.readouterr().out)

    def test_download_success_json_includes_artifact_contract_fields(self, tmp_path, capsys):
        """job download success: JSON envelope includes the artifact contract."""
        download_path = tmp_path / "results"
        download_path.mkdir()

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        data = envelope["data"]
        assert data["job_id"] == "abc123"
        assert data["download_path"] == str(download_path)
        assert data["path"] == data["download_path"]
        assert data["artifact_discovery"] == "completed"
        assert data["artifacts"] == {}
        assert set(data["missing_artifacts"]) == {"global_model", "metrics_summary", "client_logs"}

    def test_download_schema_includes_command_contract_metadata(self, capsys):
        import argparse

        from nvflare.tool.job.job_cli import cmd_job_download, def_job_cli_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        with patch("sys.argv", ["nvflare", "job", "download", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_download(MagicMock())

        assert exc_info.value.code == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["output_modes"] == ["json"]
        assert schema["streaming"] is False
        assert schema["mutating"] is True
        assert schema["idempotent"] is False
        assert schema["retry_token"] == {"supported": False}

    def test_download_default_output_dir_is_absolute_current_dir_parent(self, capsys):
        """omitted output_dir requests cwd as parent so the final path is ./<job_id>, not ./<job_id>/<job_id>."""
        args = self._make_args(job_id="abc123")

        mock_sess, _ = self._download_json(args, "/path/to/results", capsys)

        expected_dest = os.path.abspath(".")
        mock_sess.download_job_result.assert_called_once_with("abc123", expected_dest)

    def test_download_with_output_dir(self, tmp_path, capsys):
        """explicit output_dir is used as an absolute destination."""
        output_dir = tmp_path / "requested-results"
        args = self._make_args(output_dir=output_dir)

        mock_sess, _ = self._download_json(args, tmp_path / "downloaded-results", capsys)

        expected_dest = os.path.abspath(output_dir)
        mock_sess.download_job_result.assert_called_once_with("abc123", expected_dest)

    @pytest.mark.parametrize("model_name", ["FL_global_model.pt", "global_model.pt", "global_model.pth"])
    def test_download_discovers_global_model_artifact(self, tmp_path, capsys, model_name):
        """common global model filenames are reported as the global_model artifact."""
        download_path = tmp_path / "results"
        download_path.mkdir()
        model_path = download_path / model_name
        model_path.write_text("model")

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        assert envelope["data"]["artifacts"]["global_model"] == str(model_path)
        assert "global_model" not in envelope["data"]["missing_artifacts"]

    def test_download_discovers_metrics_summary_and_client_logs(self, tmp_path, capsys):
        """metrics_summary.json and client log.txt files are reported from the local download path."""
        download_path = tmp_path / "results"
        site_log_dir = download_path / "site-1"
        server_log_dir = download_path / "server"
        site_log_dir.mkdir(parents=True)
        server_log_dir.mkdir()
        metrics_path = download_path / "metrics_summary.json"
        site_log_path = site_log_dir / "log.txt"
        server_log_path = server_log_dir / "log.txt"
        metrics_path.write_text("{}")
        site_log_path.write_text("client log")
        server_log_path.write_text("server log")

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        artifacts = envelope["data"]["artifacts"]
        assert artifacts["metrics_summary"] == str(metrics_path)
        assert artifacts["client_logs"] == {"site-1": str(site_log_path)}
        assert str(server_log_path) not in artifacts["client_logs"].values()
        assert "metrics_summary" not in envelope["data"]["missing_artifacts"]
        assert "client_logs" not in envelope["data"]["missing_artifacts"]

    def test_download_client_logs_only_include_log_files(self, tmp_path, capsys):
        """nested model or metrics files must not be reported as client logs."""
        download_path = tmp_path / "results"
        site_dir = download_path / "site-1"
        site_dir.mkdir(parents=True)
        model_path = site_dir / "FL_global_model.pt"
        metrics_path = site_dir / "metrics_summary.json"
        model_path.write_text("model")
        metrics_path.write_text("{}")

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        artifacts = envelope["data"]["artifacts"]
        assert artifacts["global_model"] == str(model_path)
        assert artifacts["metrics_summary"] == str(metrics_path)
        assert "client_logs" not in artifacts
        assert "client_logs" in envelope["data"]["missing_artifacts"]

    def test_download_missing_artifacts_do_not_fail_success_response(self, tmp_path, capsys):
        """missing expected artifact categories are listed without making download fail."""
        download_path = tmp_path / "results"
        download_path.mkdir()
        metrics_path = download_path / "metrics_summary.json"
        metrics_path.write_text("{}")

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        assert envelope["data"]["artifacts"] == {"metrics_summary": str(metrics_path)}
        assert set(envelope["data"]["missing_artifacts"]) == {"global_model", "client_logs"}

    def test_download_nonexistent_path_skips_artifact_discovery(self, tmp_path, capsys):
        """nonexistent final download paths do not claim expected artifacts are missing."""
        missing_path = tmp_path / "does-not-exist"

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), missing_path, capsys)

        assert envelope["status"] == "ok"
        assert envelope["data"]["download_path"] == str(missing_path)
        assert envelope["data"]["artifact_discovery"] == "skipped"
        assert envelope["data"]["artifacts"] is None
        assert envelope["data"]["missing_artifacts"] is None

    def test_download_artifact_discovery_skips_symlink_escapes(self, tmp_path, capsys):
        """reported artifacts must stay under download_path and skip symlink escapes."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_model = outside_dir / "FL_global_model.pt"
        outside_model.write_text("outside")

        download_path = tmp_path / "results"
        download_path.mkdir()
        symlink_model = download_path / "FL_global_model.pt"
        try:
            symlink_model.symlink_to(outside_model)
        except OSError:
            pytest.skip("filesystem does not support symlinks")

        _, envelope = self._download_json(self._make_args(output_dir=tmp_path / "dest"), download_path, capsys)

        assert "global_model" not in envelope["data"]["artifacts"]
        assert "global_model" in envelope["data"]["missing_artifacts"]

    def test_download_remote_location_is_not_reported_as_local_path(self, capsys):
        """scheme-based returns are not local artifact paths."""
        _, envelope = self._download_json(self._make_args(), "https://download.example/jobs/abc123", capsys)

        assert envelope["status"] == "ok"
        assert envelope["data"]["download_path"] is None
        assert envelope["data"]["path"] == "https://download.example/jobs/abc123"
        assert envelope["data"]["artifact_discovery"] == "skipped"
        assert envelope["data"]["artifacts"] is None
        assert envelope["data"]["missing_artifacts"] is None

    def test_download_not_found_exits_1(self):
        """JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args(job_id="notfound")
        mock_sess = MagicMock()
        mock_sess.download_job_result.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_download(args)
        assert exc_info.value.code == 1

    def test_download_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.download_job_result.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_download(args)

    def test_download_connection_failed_exits_2(self, capsys):
        """NoConnection maps to CONNECTION_FAILED, exit 2."""
        from nvflare.tool.job.job_cli import cmd_job_download

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.download_job_result.side_effect = NoConnection("connection refused")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_download(args)
        assert exc_info.value.code == 2

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert envelope["exit_code"] == 2

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
