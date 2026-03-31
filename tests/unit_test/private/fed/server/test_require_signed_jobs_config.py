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

"""Tests for _require_signed_jobs config helper."""

import json
import os
from unittest.mock import MagicMock

from nvflare.private.fed.server.job_runner import _require_signed_jobs


def _make_workspace(startup_dir: str):
    ws = MagicMock()
    ws.get_startup_kit_dir.return_value = startup_dir
    return ws


class TestRequireSignedJobs:
    def test_explicit_true(self, tmp_path):
        """Explicit require_signed_jobs=true in fed_server.json → True."""
        cfg = {"require_signed_jobs": True}
        (tmp_path / "fed_server.json").write_text(json.dumps(cfg))
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is True

    def test_explicit_false(self, tmp_path):
        """Explicit require_signed_jobs=false in fed_server.json → False."""
        cfg = {"require_signed_jobs": False}
        (tmp_path / "fed_server.json").write_text(json.dumps(cfg))
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is False

    def test_key_absent_rootca_present(self, tmp_path):
        """Key absent in fed_server.json, rootCA.pem present → True (PKI inferred)."""
        (tmp_path / "fed_server.json").write_text(json.dumps({"other_key": "value"}))
        (tmp_path / "rootCA.pem").write_text("fake ca cert")
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is True

    def test_key_absent_no_rootca(self, tmp_path):
        """Key absent, no rootCA.pem → False (non-PKI inferred)."""
        (tmp_path / "fed_server.json").write_text(json.dumps({}))
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is False

    def test_no_fed_server_json_rootca_present(self, tmp_path):
        """No fed_server.json at all, rootCA.pem present → True."""
        (tmp_path / "rootCA.pem").write_text("fake ca cert")
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is True

    def test_no_fed_server_json_no_rootca(self, tmp_path):
        """No fed_server.json, no rootCA.pem → False."""
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is False

    def test_malformed_json_falls_through_to_rootca(self, tmp_path):
        """Malformed JSON in fed_server.json → falls through to rootCA heuristic."""
        (tmp_path / "fed_server.json").write_text("{invalid json")
        (tmp_path / "rootCA.pem").write_text("fake ca cert")
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is True

    def test_malformed_json_no_rootca(self, tmp_path):
        """Malformed JSON, no rootCA.pem → False."""
        (tmp_path / "fed_server.json").write_text("{invalid json")
        ws = _make_workspace(str(tmp_path))
        assert _require_signed_jobs(ws) is False

    def test_world_writable_file_still_returns_value(self, tmp_path):
        """World-writable fed_server.json: still returns correct value (with warning)."""
        cfg = {"require_signed_jobs": True}
        json_path = tmp_path / "fed_server.json"
        json_path.write_text(json.dumps(cfg))
        try:
            os.chmod(str(json_path), 0o666)  # world-writable
            ws = _make_workspace(str(tmp_path))
            assert _require_signed_jobs(ws) is True
        finally:
            os.chmod(str(json_path), 0o644)  # restore
